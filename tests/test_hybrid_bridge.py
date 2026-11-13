"""One robot, two backends: profile guards, token ownership and shared budget."""
import copy
import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from cix_harness.baxter import Baxter, build_rpc_methods, validate_config
from cix_harness.primitive_service import PrimitiveService
from cix_harness.contract import ACTION_SPEC


def config():
    cfg=json.loads(Path('config/baxter-primitives.example.json').read_text())
    cfg.update(commissioning_scope='supervised_hybrid_task', commissioned=True,
               workspace_min_m=[.3,-.6,-.2],workspace_max_m=[1.1,.6,.5])
    cfg['primitives']['calibrated']=True
    cfg['contact_guard']={'source':'operator_confirmed_fingertips_at_table',
                          'reference_eef_m':[.6,.2,-.18], 'reference_eef_xyzw':[1,0,0,0],
                          'clearance_m':.01,'radius_m':.3,'tool_extent_bound_m':.1}
    return cfg


class HybridConfiguration(unittest.TestCase):
    def test_hybrid_requires_both_geometry_profiles_and_task_limits(self):
        cfg=config();validate_config(cfg,True)
        for key in ('contact_guard','primitives'):
            invalid=copy.deepcopy(cfg);del invalid[key]
            with self.assertRaises((KeyError,ValueError)):validate_config(invalid,True)
        for changes in ({'motion_budget':65},{'lease_seconds':601},{'max_translation_m':.13}):
            with self.assertRaises(ValueError):validate_config(dict(cfg,**changes),True)
        cfg['primitives']['calibrated']=False
        with self.assertRaisesRegex(ValueError,'geometry'):validate_config(cfg,True)

    def test_health_advertises_only_explicit_hybrid_scope(self):
        r=object.__new__(Baxter);r.cfg=config();r.armed=False;r.calibrated_profile=True;r.motion_remaining=10
        result=r.health()
        self.assertEqual(result['hybrid_protocol'],'serialized_backends_v1')
        self.assertEqual(result['backend_capabilities'],['primitives','vla'])
        r.cfg['commissioning_scope']='supervised_primitive_task'
        self.assertEqual(r.health()['backend_capabilities'],['primitives'])
        self.assertIsNone(r.health()['hybrid_protocol'])

    def test_mixed_motion_obeys_both_tcp_and_contact_guards(self):
        r=object.__new__(Baxter);r.cfg=config()
        def pose(x,y,z):return {'position':NS(x=x,y=y,z=z),'orientation':NS(x=1,y=0,z=0,w=0)}
        r.guard_pose(pose(.6,.2,.1))
        with self.assertRaisesRegex(ValueError,'primitive workspace'):r.guard_pose(pose(.2,.2,.1))
        r.cfg['primitives']['tcp_workspace_min_m'][2]=-.3
        with self.assertRaisesRegex(ValueError,'primitive table'):r.guard_pose(pose(.6,.2,-.18))
        # A stricter fingertip reference still applies inside the primitive workspace.
        r.cfg['contact_guard']['reference_eef_m'][2]=.12
        with self.assertRaisesRegex(ValueError,'Fingertip'):r.guard_pose(pose(.6,.2,.1))


class TokenOwnership(unittest.TestCase):
    def setUp(self):
        self.robot=Mock();self.robot.cfg=config();self.robot.motion_remaining=10
        self.robot.pending=('old-policy',time.monotonic(),{})
        self.service=Mock();self.service.observation=('old-primitive',);self.service.prepared=('old-plan',)
        def observe():
            self.robot.pending=('new-policy',time.monotonic(),{})
            return {'id':'new-policy'}
        self.robot.observe.side_effect=observe
        def primitive_observe():
            observe();self.service.observation=('new-primitive',);self.service.prepared=None
            return {'id':'new-primitive'}
        self.service.observe.side_effect=primitive_observe
        with patch('cix_harness.primitive_service.PrimitiveService',return_value=self.service):
            self.methods=build_rpc_methods(self.robot)

    def test_primitive_observation_cannot_be_consumed_as_policy_observation(self):
        result=self.methods['primitive.observe']()
        self.assertEqual(result['id'],'new-primitive');self.assertIsNone(self.robot.pending)
        self.assertEqual(self.service.observation,('new-primitive',))

    def test_policy_observation_invalidates_primitive_observation_and_prepared_plan(self):
        self.methods['env.observe']()
        self.assertIsNone(self.service.observation);self.assertIsNone(self.service.prepared)
        self.assertEqual(self.robot.pending[0],'new-policy')

    def test_every_policy_operation_invalidates_old_primitive_plan_even_on_failure(self):
        for name in ('env.preview','env.inspect','env.step','env.step_chunk','env.preview_segment','env.step_segment','env.scene_snapshot'):
            self.service.observation=('old',);self.service.prepared=('old',)
            self.methods[name]('arbitrary')
            self.assertIsNone(self.service.observation);self.assertIsNone(self.service.prepared)
        self.service.prepared=('old',);self.robot.step.side_effect=RuntimeError('failure before dispatch')
        with self.assertRaises(RuntimeError):self.methods['env.step']('arbitrary')
        self.assertIsNone(self.service.prepared)

    def test_primitive_prepare_and_execute_invalidate_policy_tokens_even_on_failure(self):
        for name,target in [('primitive.prepare',self.service.prepare),('primitive.execute',self.service.execute)]:
            self.robot.pending=('old-policy',);target.side_effect=ValueError('invalid token')
            with self.assertRaises(ValueError):self.methods[name]('old')
            self.assertIsNone(self.robot.pending)

    def test_health_does_not_invalidate_preflight(self):
        self.methods['healthz']()
        self.assertIsNotNone(self.service.prepared);self.assertIsNotNone(self.robot.pending)


class SharedExecutionState(unittest.TestCase):
    def test_hybrid_windows_and_segments_still_require_new_unconsumed_observations(self):
        r=object.__new__(Baxter);r.cfg=config();r.calibrated_profile=True;r.pending=None
        with self.assertRaisesRegex(ValueError,'unconsumed'):
            r.step_chunk('old',ACTION_SPEC,[[0]*7])
        with patch.dict('sys.modules',{'tf':NS(transformations=Mock())}):
            with self.assertRaisesRegex(ValueError,'unconsumed'):
                r.prepare_segment('old',ACTION_SPEC,[[0]*7])

    def test_failed_new_observation_removes_old_primitive_observation(self):
        r=Mock();r.cfg=config();r.observe.side_effect=RuntimeError('camera unavailable')
        service=PrimitiveService(r);service.observation=('old',);service.prepared=('old',)
        with self.assertRaisesRegex(RuntimeError,'camera'):service.observe()
        self.assertIsNone(service.observation);self.assertIsNone(service.prepared)

    def test_primitive_execution_debits_same_robot_budget_and_checks_lease_ready(self):
        r=Mock();r.cfg=config();r.motion_remaining=3;r.armed=True
        r.joints.return_value={'j':0};r.endpoint_position.return_value=[.6,.2,.1]
        r.grip.command_position.return_value=True;r.grip.position.return_value=100.;r.grip.gripping.return_value=False
        service=PrimitiveService(r)
        service.prepared=('p',time.monotonic(),{'j':0},[{'kind':'gripper','open':True}],[None],1,'open',False)
        with patch('cix_harness.primitive_service.time.sleep'):
            receipt=service.execute('p')
        self.assertTrue(receipt['applied']);self.assertEqual(r.motion_remaining,2)
        service.prepared=('q',time.monotonic(),{'j':0},[{'kind':'gripper','open':True}],[None],1,'open',False)
        r.ready.side_effect=RuntimeError('Motion lease expired')
        with self.assertRaisesRegex(RuntimeError,'lease'):service.execute('q')
        self.assertEqual(r.motion_remaining,2)
        self.assertEqual(r.grip.command_position.call_count,1)

    def test_primitive_only_scope_still_excludes_policy_rpc(self):
        r=Mock();r.cfg=config();r.cfg['commissioning_scope']='supervised_primitive_task'
        methods=build_rpc_methods(r)
        self.assertIn('primitive.execute',methods);self.assertNotIn('env.step',methods)


if __name__=='__main__':unittest.main()
