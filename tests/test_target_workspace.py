"""Only explicit perception targets get recoverable workspace rejections."""
import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from cix_harness.primitives import TargetWorkspaceError, check_tcp
from cix_harness.primitive_service import PrimitiveService


class TargetWorkspace(unittest.TestCase):
    def service(self,held=True):
        p=json.loads(Path('config/baxter-primitives.example.json').read_text())['primitives']
        p['tcp_workspace_min_m'][2]=-.17
        robot=Mock();robot.cfg={'primitives':p,'max_observation_age_s':3};robot.motion_remaining=64
        robot.joints.return_value={'j':0.}
        robot.limb.endpoint_pose.return_value={'position':NS(x=.6,y=.2,z=.1),
                                              'orientation':NS(x=1.,y=0.,z=0.,w=0.)}
        service=PrimitiveService(robot)
        state={'eef_m':[.6,.2,.1],'eef_xyzw':[1,0,0,0],'gripping':held,
               'joint_names':['j'],'joint_rad':[0.]}
        service.observation=({'id':'current','state':state},time.monotonic(),None)
        geometry=NS(PoseStamped=lambda:NS(header=NS(),pose=NS(position=NS(),orientation=NS())))
        modules={'geometry_msgs.msg':geometry,'tf':NS(transformations=NS(quaternion_slerp=lambda a,b,f:a))}
        return service,robot,modules

    def assert_no_motion(self,service,robot):
        robot.move_segment.assert_not_called();robot.grip.command_position.assert_not_called()
        robot.continuous_ik.solve.assert_not_called()
        self.assertIsNone(service.prepared);self.assertIsNone(service.observation)
        self.assertEqual(robot.motion_remaining,64)

    def test_outside_place_target_is_recoverable_before_any_ik_or_motion(self):
        service,robot,modules=self.service()
        with patch.dict('sys.modules',modules):
            result=service.prepare('current','place',target_base_m=[1.5,.2,-.12])
        self.assertEqual(result['failure_code'],'target_workspace')
        self.assertTrue(result['recoverable']);self.assertFalse(result['applied'])
        self.assertEqual(result['status'],'rejected')
        self.assert_no_motion(service,robot)

    def test_depth_grounded_outside_pick_target_is_also_recoverable(self):
        service,robot,modules=self.service(held=False)
        with patch.dict('sys.modules',modules), \
             patch('cix_harness.primitive_service.ground_pixel',return_value=[1.5,.2,-.12]):
            result=service.prepare('current','pick',uv=[.8,.5])
        self.assertEqual(result['failure_code'],'target_workspace')
        self.assert_no_motion(service,robot)

    def test_table_violation_stays_hard_even_when_xy_is_outside(self):
        service,robot,modules=self.service()
        with patch.dict('sys.modules',modules),self.assertRaisesRegex(ValueError,'below table') as raised:
            service.prepare('current','place',target_base_m=[1.5,.2,-.25])
        self.assertNotIsInstance(raised.exception,TargetWorkspaceError)
        self.assert_no_motion(service,robot)

    def test_generated_hover_outside_workspace_is_not_a_target_rejection(self):
        service,robot,modules=self.service(held=False)
        with patch.dict('sys.modules',modules),self.assertRaisesRegex(ValueError,'outside calibrated') as raised:
            service.prepare('current','pick',target_base_m=[.7,.2,.4])
        self.assertNotIsInstance(raised.exception,TargetWorkspaceError)
        self.assert_no_motion(service,robot)

    def test_invalid_configuration_stays_hard(self):
        service,robot,modules=self.service()
        service.profile['tcp_workspace_min_m'][0]=2
        with patch.dict('sys.modules',modules),self.assertRaisesRegex(ValueError,'Unordered workspace') as raised:
            service.prepare('current','place',target_base_m=[1.5,.2,-.12])
        self.assertNotIsInstance(raised.exception,TargetWorkspaceError)
        self.assert_no_motion(service,robot)

    def test_interpolated_path_outside_workspace_stays_hard(self):
        service,robot,modules=self.service(held=False)
        with patch.dict('sys.modules',modules), \
             patch('cix_harness.primitives.tcp_from_eef',side_effect=[[.6,.2,.1],[1.2,.2,.1]]), \
             self.assertRaisesRegex(ValueError,'Interpolated TCP leaves workspace') as raised:
            service.prepare('current','pick',target_base_m=[.7,.25,-.12])
        self.assertNotIsInstance(raised.exception,TargetWorkspaceError)
        self.assert_no_motion(service,robot)


if __name__=='__main__':unittest.main()
