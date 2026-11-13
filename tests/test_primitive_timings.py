"""Elapsed component costs remain useful on refusal, partial failure and success."""
import json
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from cix_harness.kinematics import IKUnreachableError
from cix_harness.primitive_service import PrimitiveService
from cix_harness.primitives import TargetDepthError, plan


class Clock:
    def __init__(self):self.value=100.
    def now(self):return self.value
    def advance(self,seconds):self.value+=seconds
    def work(self,seconds,value=None,error=None):
        def invoke(*args,**kwargs):
            self.advance(seconds)
            if error is not None:raise error
            return value
        return invoke


class PrimitiveTimings(unittest.TestCase):
    def setUp(self):
        self.clock=Clock()
        self.monotonic=patch('cix_harness.primitive_service.time.monotonic',self.clock.now)
        self.sleep=patch('cix_harness.primitive_service.time.sleep',self.clock.advance)
        self.monotonic.start();self.sleep.start()
        self.addCleanup(self.monotonic.stop);self.addCleanup(self.sleep.stop)
        p=json.loads(Path('config/baxter-primitives.example.json').read_text())['primitives']
        p['tcp_workspace_min_m'][2]=-.17;p['calibrated']=True
        r=Mock();r.cfg={'primitives':p,'commissioning_scope':'supervised_primitive_task',
                      'max_observation_age_s':3,'arrival_position_tolerance_m':.01}
        r.armed=True;r.motion_remaining=24;r.joints.return_value={'j':0.}
        r.endpoint_position.return_value=[.6,.2,.1]
        r.limb.endpoint_pose.return_value={'position':NS(x=.6,y=.2,z=.1),'orientation':NS(x=1.,y=0.,z=0.,w=0.)}
        r.continuous_ik.solve.return_value={'j':0.}
        r.grip.gripping.return_value=False;r.grip.position.return_value=100.
        r.grip.command_position.return_value=True;r.grip.stop.return_value=True
        r.arrival_error.return_value=(0.,0.)
        self.state={'eef_m':[.6,.2,.1],'eef_xyzw':[1,0,0,0],'gripping':False,
                    'joint_names':['j'],'joint_rad':[0.]}
        self.r=r;self.service=PrimitiveService(r)
        self.service.observation=({'id':'o','state':self.state},self.clock.now(),None)
        geometry=NS(PoseStamped=lambda:NS(header=NS(),pose=NS(position=NS(),orientation=NS())))
        modules={'geometry_msgs.msg':geometry,'tf':NS(transformations=NS(quaternion_slerp=lambda a,b,f:a)),
                 'pyrealsense2':NS(rs2_project_point_to_pixel=Mock())}
        self.modules=patch.dict('sys.modules',modules);self.modules.start();self.addCleanup(self.modules.stop)

    def events(self,name):
        return [call.kwargs for call in self.r.audit.call_args_list if call.args[0]==name]

    def prepared(self,stages):
        paths=[[({},NS())] if s['kind']=='move' else None for s in stages]
        self.service.prepared=('p',self.clock.now(),{'j':0.},stages,paths,
                               sum(s['kind']!='verify_grasp' for s in stages),'pick',False)

    def test_observation_preserves_camera_cost_without_double_counting_outer_total(self):
        obs={'id':'fresh','state':self.state,'timings_s':{'camera_total':1.5}}
        self.r.observe.side_effect=self.clock.work(2.,obs)
        self.r.camera.last_depth=NS(copy=lambda:'copy')
        with patch('cix_harness.primitive_service.workspace_metadata',side_effect=self.clock.work(.3,{})):
            result=self.service.observe()
        timing=result['timings_s']
        self.assertAlmostEqual(timing['camera_total'],1.5)
        self.assertAlmostEqual(timing['robot_observe_total'],2.)
        self.assertAlmostEqual(timing['observation_metadata'],.3)
        self.assertAlmostEqual(timing['observe_total'],2.3)
        self.assertEqual(self.events('primitive_observed')[0]['timings_s'],timing)
        self.r.move_segment.assert_not_called();self.r.grip.command_position.assert_not_called()

    def test_observation_failure_keeps_elapsed_cost_and_invalidates_prior_tokens(self):
        self.service.prepared=('old',)
        self.r.observe.side_effect=self.clock.work(1.2,error=RuntimeError('camera failed'))
        with self.assertRaisesRegex(RuntimeError,'camera failed'):self.service.observe()
        event=self.events('primitive_observe_failed')[0]
        self.assertAlmostEqual(event['timings_s']['robot_observe_total'],1.2)
        self.assertAlmostEqual(event['timings_s']['observe_total'],1.2)
        self.assertIsNone(self.service.prepared);self.assertIsNone(self.service.observation)
        self.assertFalse(self.events('primitive_observed'))

    def test_grounding_and_complete_ik_are_separate_from_plan_total(self):
        self.r.continuous_ik.solve.side_effect=self.clock.work(.01,{'j':0.})
        def planning(*args,**kwargs):
            self.clock.advance(.2)
            return plan(*args,**kwargs)
        with patch('cix_harness.primitive_service.ground_pixel',side_effect=self.clock.work(.1,[.7,.25,-.12])), \
             patch('cix_harness.primitive_service.plan',side_effect=planning):
            result=self.service.prepare('o','pick',uv=[.5,.5])
        timing=result['timings_s'];ik_seconds=.01*self.r.continuous_ik.solve.call_count
        self.assertEqual(result['status'],'ready')
        self.assertAlmostEqual(timing['grounding'],.1)
        self.assertAlmostEqual(timing['skill_plan'],.2)
        self.assertAlmostEqual(timing['ik_preflight'],ik_seconds)
        self.assertAlmostEqual(timing['prepare_total'],.3+ik_seconds)
        self.r.move_segment.assert_not_called()

    def test_grounding_and_ik_rejections_keep_spent_time(self):
        with patch('cix_harness.primitive_service.ground_pixel',side_effect=self.clock.work(.4,error=TargetDepthError('hole'))):
            result=self.service.prepare('o','pick',uv=[.5,.5])
        self.assertEqual(result['failure_code'],'target_depth')
        self.assertAlmostEqual(result['timings_s']['grounding'],.4)
        self.assertEqual(result['timings_s']['ik_preflight'],0.)
        self.service.observation=({'id':'o','state':self.state},self.clock.now(),None)
        self.r.continuous_ik.solve.side_effect=self.clock.work(.7,error=IKUnreachableError('no solution'))
        result=self.service.prepare('o','pick',target_base_m=[.7,.25,-.12])
        self.assertEqual(result['failure_code'],'ik_preflight')
        self.assertAlmostEqual(result['timings_s']['ik_preflight'],.7)
        self.assertAlmostEqual(self.events('primitive_prepare_rejected')[-1]['timings_s']['prepare_total'],.7)
        self.r.grip.command_position.assert_not_called()

    def test_stage_budget_refusal_has_timings_and_does_not_run_ik(self):
        self.r.motion_remaining=0
        result=self.service.prepare('o','pick',target_base_m=[.7,.25,-.12])
        self.assertEqual(result['failure_code'],'stage_budget')
        self.assertEqual(result['timings_s']['ik_preflight'],0.)
        self.assertIn('prepare_total',result['timings_s'])
        self.assertEqual(self.events('primitive_prepare_rejected')[0]['failure_code'],'stage_budget')
        self.r.continuous_ik.solve.assert_not_called()

    def test_stage_receipts_separate_motion_command_settle_and_verification(self):
        self.prepared([{'kind':'move'},{'kind':'gripper','open':True},{'kind':'verify_grasp','held':False}])
        self.r.move_segment.side_effect=self.clock.work(1.4)
        self.r.arrival_error.side_effect=self.clock.work(.1,(0.,0.))
        self.r.grip.command_position.side_effect=self.clock.work(.2,True)
        result=self.service.execute('p');timing=result['timings_s']
        self.assertEqual(result['status'],'completed')
        self.assertAlmostEqual(timing['motion'],1.4)
        self.assertAlmostEqual(timing['motion_verification'],.1)
        self.assertAlmostEqual(timing['gripper_command'],.2)
        self.assertAlmostEqual(timing['gripper_settle'],.25)
        self.assertAlmostEqual(timing['execute_total'],1.95)
        self.assertAlmostEqual(result['completed_stages'][0]['duration_s'],1.5)
        self.assertAlmostEqual(result['completed_stages'][1]['duration_s'],.45)
        self.assertIn('grasp_verification',result['completed_stages'][2]['timings_s'])
        self.assertEqual(len(self.events('primitive_stage_applied')),3)
        self.assertEqual(self.events('primitive_completed')[0]['timings_s'],timing)

    def test_unknown_motion_failure_logs_elapsed_failed_stage_not_completion(self):
        self.prepared([{'kind':'move'},{'kind':'gripper','open':True}])
        self.r.move_segment.side_effect=self.clock.work(.8,error=RuntimeError('motor link lost'))
        self.r.grip.stop.side_effect=self.clock.work(.1,True)
        with self.assertRaisesRegex(RuntimeError,'motor link lost'):self.service.execute('p')
        failed=self.events('primitive_failed')[0]
        self.assertEqual(failed['completed_stages'],0)
        self.assertEqual(failed['failed_stage']['status'],'failed')
        self.assertAlmostEqual(failed['failed_stage']['duration_s'],.8)
        self.assertAlmostEqual(failed['failed_stage']['timings_s']['motion'],.8)
        self.assertAlmostEqual(failed['timings_s']['execute_total'],.9)
        self.assertAlmostEqual(failed['timings_s']['stop'],.1)
        self.assertFalse(self.events('primitive_completed'));self.assertFalse(self.events('primitive_stage_applied'))
        self.r.grip.command_position.assert_not_called()
        self.assertIsNone(self.service.prepared)

    def test_gripper_timeout_records_command_duration_without_settle_or_success(self):
        self.prepared([{'kind':'gripper','open':True}])
        self.r.grip.command_position.side_effect=self.clock.work(.6,False)
        with self.assertRaisesRegex(RuntimeError,'timeout'):self.service.execute('p')
        failed=self.events('primitive_failed')[0]
        self.assertAlmostEqual(failed['failed_stage']['timings_s']['gripper_command'],.6)
        self.assertEqual(failed['timings_s']['gripper_settle'],0.)
        self.assertFalse(self.events('primitive_stage_applied'))

    def test_execution_rejection_logs_timing_without_dispatch_or_stop(self):
        self.prepared([{'kind':'gripper','open':True}]);self.r.armed=False
        with self.assertRaisesRegex(RuntimeError,'read-only'):self.service.execute('p')
        self.assertIn('execute_total',self.events('primitive_execute_failed')[0]['timings_s'])
        self.assertFalse(self.events('primitive_dispatch'))
        self.r.grip.command_position.assert_not_called();self.r.grip.stop.assert_not_called()


if __name__=='__main__':unittest.main()
