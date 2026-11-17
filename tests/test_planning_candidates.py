import copy,json,math,time,unittest
from pathlib import Path
from unittest.mock import Mock,patch
from cix_harness.planning import candidate_profiles,stage_phase
from cix_harness.primitive_service import PrimitiveService,IKPreflightError
from cix_harness.collision import Capsule,KDLArmCapsules,ArmCollisionError


class PlanningCandidates(unittest.TestCase):
    def test_symmetric_grasp_prefers_small_rotation_and_preserves_profile(self):
        p={'tool_orientation_xyzw':[1,0,0,0]}
        rows=list(candidate_profiles(p,{'eef_xyzw':[0,1,0,0],'gripping':False},'pick',{'grasp_yaw_offsets_rad':[0,math.pi]}))
        self.assertAlmostEqual(rows[0][0],math.pi)
        self.assertEqual(p['tool_orientation_xyzw'],[1,0,0,0])
        with self.assertRaises(ValueError):list(candidate_profiles(p,{'eef_xyzw':[1,0,0,0]},'pick',{'grasp_yaw_offsets_rad':[float('nan')]}))

    def test_preflight_failure_tries_next_yaw_before_any_motor_call(self):
        p=json.loads(Path('config/baxter-primitives.example.json').read_text())['primitives']
        p['tcp_workspace_min_m'][2]=-.17
        r=Mock();r.cfg={'primitives':p,'max_observation_age_s':3,'planning':{'grasp_yaw_offsets_rad':[0,math.pi]}}
        r.motion_remaining=20;r.joints.return_value={'j':0.}
        s=PrimitiveService(r)
        state={'eef_m':[.6,.2,.1],'eef_xyzw':[1,0,0,0],'gripping':False,'joint_names':['j'],'joint_rad':[0.]}
        s.observation=({'id':'obs','state':state},time.monotonic(),None)
        s._preflight=Mock(side_effect=[IKPreflightError('wrong elbow branch'),({'j':0.},[])])
        result=s.prepare('obs','pick',target_base_m=[.7,.2,-.1])
        self.assertEqual(result['status'],'ready')
        self.assertEqual([c['status'] for c in result['planning_candidates']],['rejected','ready'])
        r.move_segment.assert_not_called();r.grip.command_position.assert_not_called()

    def test_head_collision_checked_independently_of_other_arm(self):
        m=object.__new__(KDLArmCapsules);m.cfg={'head_minimum_distance_m':.04}
        m.head_capsules=[Capsule('head',(0,0,0),(0,0,0),.22)]
        m.capsules=lambda q,holding:[Capsule('elbow',(q['j'],0,0),(q['j'],0,.1),.06)]
        self.assertGreater(m.head_clearance({'j':.5}),.04)
        with self.assertRaises(ArmCollisionError):m.head_clearance({'j':.25})

    def test_failure_to_lift_is_not_classified_as_transit(self):
        stages=[{'kind':'gripper','open':False},{'kind':'move','tcp_m':[.5,0,.1]}]
        self.assertEqual(stage_phase(stages,1,'pick'),'lift')
