import json
import math
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
from cix_harness.primitives import plan, tcp_from_eef, eef_from_tcp, validate_profile
from cix_harness.primitive_service import PrimitiveService
from cix_harness.primitive_runner import run
from cix_harness.kinematics import IKUnreachableError


def profile():
    p=json.loads(Path('config/baxter-primitives.example.json').read_text())['primitives']
    p['tcp_workspace_min_m'][2]=-.17
    return p

def state(held=False):
    return {'eef_m':[.6,.2,.1],'eef_xyzw':[1,0,0,0],'gripping':held}

class Plans(unittest.TestCase):
    def test_pick_stages_and_no_fixed_depth(self):
        p=profile();stages=plan('pick',state(),p,[.7,.25,-.12])
        contacts=[s['tcp_m'][2] for s in stages if s['kind']=='move']
        self.assertAlmostEqual(min(contacts),-.135)
        self.assertEqual([s['open'] for s in stages if s['kind']=='gripper'],[True,False])
        self.assertEqual(stages[-1],{'kind':'verify_grasp','held':True})
    def test_place_uses_object_bottom_offset_and_retracts(self):
        stages=plan('stack',state(True),profile(),[.7,.25,-.12])
        self.assertAlmostEqual(stages[3]['tcp_m'][2],-.1)
        self.assertEqual(stages[4],{'kind':'gripper','open':True})
        self.assertEqual(stages[5],{'kind':'verify_grasp','held':False})
        self.assertGreater(stages[6]['tcp_m'][2],stages[3]['tcp_m'][2])
    def test_rejects_wrong_grasp_state_workspace_and_table(self):
        for skill,st,target in [('pick',state(True),[.7,.2,0]),('place',state(),[.7,.2,0]),
                                ('pick',state(),[1.5,.2,0]),('pick',state(),[.7,.2,-.18])]:
            with self.assertRaises(ValueError):plan(skill,st,profile(),target)
    def test_tcp_offset_rotates_with_tool(self):
        p=profile();p['tcp_offset_tool_m']=[0,0,.1]
        self.assertEqual(tcp_from_eef([0,0,0],[1,0,0,0],p),[0.,0.,-.1])
        self.assertEqual(eef_from_tcp([0,0,-.1],[1,0,0,0],p),[0.,0.,0.])
    def test_nonfinite_configuration_rejected(self):
        p=profile();p['transit_z_m']=math.nan
        with self.assertRaises(ValueError):validate_profile(p)

class Execution(unittest.TestCase):
    def service(self,stages,held=False):
        r=Mock();r.cfg={'primitives':profile(),'commissioning_scope':'supervised_primitive_task',
                       'max_observation_age_s':3,'arrival_position_tolerance_m':.01}
        r.cfg['primitives']['calibrated']=True
        r.armed=True;r.motion_remaining=20;r.joints.return_value={'j':0.};r.endpoint_position.return_value=[.6,.2,.1]
        r.grip.gripping.return_value=held;r.grip.command_position.return_value=True;r.grip.position.return_value=100.
        r.arrival_error.return_value=(0.,0.)
        s=PrimitiveService(r)
        s.prepared=('token',time.monotonic(),{'j':0.},stages,[[({},NS())] if x['kind']=='move' else None for x in stages],
                    sum(x['kind']!='verify_grasp' for x in stages),'pick',held)
        return s,r
    def test_readonly_and_duplicate_plan_never_actuate(self):
        s,r=self.service([{'kind':'gripper','open':True}]);r.armed=False
        with self.assertRaises(RuntimeError):s.execute('token')
        r.grip.command_position.assert_not_called()
        with self.assertRaises(ValueError):s.execute('token')
    def test_empty_grasp_prevents_lift(self):
        s,r=self.service([{'kind':'gripper','open':False},{'kind':'verify_grasp','held':True},{'kind':'move'}])
        receipt=s.execute('token')
        self.assertEqual(receipt['failure_code'],'empty_grasp')
        self.assertTrue(receipt['stopped']);self.assertEqual(receipt['status'],'failed')
        r.move_segment.assert_not_called();r.grip.stop.assert_called_once()
    def test_ik_tracking_failure_prevents_release(self):
        s,r=self.service([{'kind':'move'},{'kind':'gripper','open':True}],True)
        r.arrival_error.return_value=(.03,0.)
        with self.assertRaisesRegex(RuntimeError,'reach target'):s.execute('token')
        r.grip.command_position.assert_not_called()
    def test_failed_stop_never_returns_recoverable_empty_grasp(self):
        s,r=self.service([{'kind':'gripper','open':False},{'kind':'verify_grasp','held':True}])
        r.limb.set_joint_positions.side_effect=RuntimeError('Stop unavailable')
        with self.assertRaisesRegex(RuntimeError,'Empty grasp'):s.execute('token')
        with self.assertRaises(ValueError):s.execute('token')
    def test_object_lost_after_lift_is_not_reported_as_empty_grasp(self):
        s,r=self.service([{'kind':'verify_grasp','held':True},{'kind':'verify_grasp','held':True}],True)
        r.grip.gripping.side_effect=[True,True,False]
        with self.assertRaisesRegex(RuntimeError,'lost'):s.execute('token')
    def test_success_and_no_replay(self):
        s,r=self.service([{'kind':'gripper','open':True}])
        self.assertTrue(s.execute('token')['applied'])
        with self.assertRaises(ValueError):s.execute('token')
        self.assertEqual(r.grip.command_position.call_count,1)
    def test_stale_plan_does_not_dispatch(self):
        s,r=self.service([{'kind':'gripper','open':True}]);v=list(s.prepared);v[1]-=10;s.prepared=tuple(v)
        with self.assertRaisesRegex(RuntimeError,'expired'):s.execute('token')
        r.grip.command_position.assert_not_called()

class Runner(unittest.TestCase):
    def args(self,execute=True):
        return NS(task='抓起方块',execute=execute,vlm_port=9110,vlm_model='local',robot_port=9102,max_turns=3,task_kind='pick')
    def test_complete_loop_does_not_use_vla(self):
        calls=[]
        def call(method,args=(),**kw):
            calls.append(method)
            if method=='healthz':return {'motion_scope':'supervised_primitive_task','armed':True}
            if method=='primitive.observe':return {'id':'obs','image':'test','state':{'gripping':calls.count('primitive.execute')>0}}
            if method=='primitive.prepare':return {'plan_id':'p'}
            if method=='primitive.execute':return {'plan_id':'p','applied':True}
            raise AssertionError(method)
        robot=NS(call=call);planner=Mock(side_effect=[{'decision':'act','reason':'none','skill':'pick','uv':[.5,.5],'evidence':'cube'},
                                                   {'decision':'done','reason':'none','skill':'','uv':None,'evidence':'lifted'}])
        self.assertEqual(run(self.args(),Mock(),robot,planner),0)
        self.assertEqual(calls.count('primitive.execute'),1)
    def test_lost_reply_never_retries(self):
        robot=Mock()
        robot.call.side_effect=[{'motion_scope':'supervised_primitive_task','armed':True},
                                {'id':'obs','image':'test','state':{'gripping':False}}, {'plan_id':'p'},TimeoutError('lost')]
        planner=Mock(return_value={'decision':'act','reason':'none','skill':'pick','uv':[.5,.5],'evidence':'cube'})
        with self.assertRaises(TimeoutError):run(self.args(),Mock(),robot,planner)
        self.assertEqual(sum(c.args[0]=='primitive.execute' for c in robot.call.call_args_list),1)

class Preflight(unittest.TestCase):
    def test_full_ik_failure_precedes_every_actuator_operation(self):
        import sys
        from unittest.mock import patch
        geometry=NS(PoseStamped=lambda:NS(header=NS(),pose=NS(position=NS(),orientation=NS())))
        tf=NS(transformations=NS(quaternion_slerp=lambda a,b,f:a))
        p=profile();r=Mock();r.cfg={'primitives':p,'max_observation_age_s':3}
        r.motion_remaining=64;r.joints.return_value={'j':0.}
        r.limb.endpoint_pose.return_value={'position':NS(x=.6,y=.2,z=.1),'orientation':NS(x=1.,y=0.,z=0.,w=0.)}
        r.continuous_ik.solve.side_effect=[{'j':0.},IKUnreachableError('No continuous solution')]
        service=PrimitiveService(r)
        st=state();st.update(joint_names=['j'],joint_rad=[0.])
        service.observation=({'id':'o','state':st},time.monotonic(),None)
        with patch.dict(sys.modules,{'geometry_msgs':NS(msg=geometry),'geometry_msgs.msg':geometry,'tf':tf}):
            result=service.prepare('o','pick',target_base_m=[.7,.25,-.12])
            self.assertEqual(result['failure_code'],'ik_preflight');self.assertFalse(result['applied'])
        r.move_segment.assert_not_called();r.grip.command_position.assert_not_called()
        self.assertIsNone(service.prepared)
        self.assertIsNone(service.observation)

    def test_successful_full_plan_is_still_read_only(self):
        import sys
        from unittest.mock import patch
        geometry=NS(PoseStamped=lambda:NS(header=NS(),pose=NS(position=NS(),orientation=NS())))
        tf=NS(transformations=NS(quaternion_slerp=lambda a,b,f:a))
        p=profile();r=Mock();r.cfg={'primitives':p,'max_observation_age_s':3}
        r.motion_remaining=64;r.joints.return_value={'j':0.}
        r.limb.endpoint_pose.return_value={'position':NS(x=.6,y=.2,z=.1),'orientation':NS(x=1.,y=0.,z=0.,w=0.)}
        r.continuous_ik.solve.return_value={'j':0.}
        service=PrimitiveService(r);st=state();st.update(joint_names=['j'],joint_rad=[0.])
        service.observation=({'id':'o','state':st},time.monotonic(),None)
        with patch.dict(sys.modules,{'geometry_msgs':NS(msg=geometry),'geometry_msgs.msg':geometry,'tf':tf}):
            result=service.prepare('o','pick',target_base_m=[.7,.25,-.12])
        self.assertTrue(result['ik_preflight']);self.assertFalse(result['applied'])
        r.move_segment.assert_not_called();r.grip.command_position.assert_not_called()
        self.assertGreater(r.continuous_ik.solve.call_count,10)
