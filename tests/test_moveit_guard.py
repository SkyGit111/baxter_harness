import json
import math
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from cix_harness.moveit_guard import MoveItCollisionError, options, segment_samples
from cix_harness.moveit_probe import joint_names
from cix_harness.primitive_service import PrimitiveService
from cix_harness.trajectory import JointTrajectory, motion_limits


class MoveItPrimitivePath(unittest.TestCase):
    def setUp(self):
        self.r = Mock()
        self.r.cfg = json.loads(Path('config/baxter-primitives.example.json').read_text())
        self.r.cfg.update(arm='right', commissioned=True, max_observation_age_s=3,
                          planning={'collision_backend': 'moveit'})
        self.r.cfg['primitives']['calibrated'] = True
        self.r.armed = True
        self.r.motion_remaining = 20
        self.start = dict.fromkeys(sorted(joint_names('right')), 0.)
        self.end = dict(self.start, right_s0=.08)
        self.knots = [(self.end, None)]
        self.full = dict(self.start, **dict.fromkeys(joint_names('left'), .33), head_pan=.12)
        self.r.joints.return_value = self.start
        self.r.full_body_joint_sample.side_effect = lambda: (time.monotonic(), self.full)
        self.service = PrimitiveService(self.r)
        self.guard = self.service.moveit
        self.guard.ros = object()
        self.guard.validity = Mock()
        self.scene_state = NS(joint_state=NS(name=list(self.full)+['finger'], position=[-.9]*16))
        self.scene_report = {'world_objects': {}, 'attached_objects': []}
        self.guard._scene = Mock(return_value=(self.scene_state, self.scene_report))
        self.seen = []

        def validate(state):
            q = dict(zip(state.joint_state.name, state.joint_state.position))
            self.seen.append(q)
            return NS(valid=True, contacts=[])
        self.guard._valid = Mock(side_effect=validate)

    def test_complete_curve_and_measured_inactive_arm_enter_each_query(self):
        evidence = self.guard.check(self.start, [(0, 'transit', self.knots)])
        self.assertGreater(evidence['samples'], 2)
        self.assertEqual(self.seen[0]['right_s0'], 0.)
        self.assertEqual(self.seen[-1]['right_s0'], .08)
        self.assertTrue(all(q['left_s1'] == .33 and q['head_pan'] == .12 and q['finger'] == -.9
                            for q in self.seen))
        self.assertEqual(self.guard._scene.call_count, 2)

    def test_execution_reference_belongs_to_last_successfully_checked_segment(self):
        segments=[(0,'transit',self.knots)]
        self.guard.check(self.start,segments,executing=True)
        self.full=dict(self.full,left_s1=.342)
        self.guard.check(self.start,segments,executing=True)
        checked=self.guard.execution_reference
        self.assertEqual(checked['left_s1'],.342)
        self.full=dict(self.full,left_s1=.354)
        self.guard._valid=Mock(return_value=NS(valid=False,contacts=[]))
        with self.assertRaisesRegex(MoveItCollisionError,'rejected stage'):
            self.guard.check(self.start,segments,executing=True)
        self.assertIs(self.guard.execution_reference,checked)

    def test_far_peer_motion_with_metric_certificate_still_checks_actual_body_and_head(self):
        self.guard.check(self.start,[(0,'transit',self.knots)],executing=True)
        self.full=dict(self.full,left_s1=.39,right_s0=.05)
        certificate=Mock(return_value=True)
        with patch.object(type(self.r),'peer_path_supports',certificate,create=True):
            self.guard.check_inactive(self.guard.execution_reference,during_motion=True)
            self.assertEqual(self.seen[-1]['left_s1'],.39)
            self.assertEqual(self.seen[-1]['right_s0'],.05)
            self.guard._valid=Mock(return_value=NS(valid=False,contacts=[]))
            with self.assertRaises(MoveItCollisionError):
                self.guard.check_inactive(self.guard.execution_reference,during_motion=True)
            self.full=dict(self.full,head_pan=.30)
            with self.assertRaisesRegex(RuntimeError,'Head moved'):
                self.guard.check_inactive(self.guard.execution_reference,during_motion=True)

    def test_continuous_points_are_from_actual_cubic_executor(self):
        self.r.cfg['planning']['trajectory_mode'] = 'continuous'
        knots = [dict(self.start, right_s0=.04, right_w0=.09), dict(self.start, right_s0=.10, right_w0=.11)]
        speed, accel = motion_limits(self.r.cfg, 'transit')
        expected = list(JointTrajectory(self.start, knots, speed, accel*.5).sampled_positions(.01))
        actual = list(segment_samples(self.start, [(q, None) for q in knots], self.r.cfg, 'transit'))
        self.assertEqual(actual, expected)
        self.assertTrue(any(abs(q['right_s0']/.04 - q['right_w0']/.09) > .0001
                            for q in actual if 0 < q['right_s0'] < .04))

    def test_invalid_interior_sample_blocks_token_and_dispatch(self):
        state = {'eef_m': [.6,.2,.1], 'eef_xyzw': [1,0,0,0], 'gripping': False}
        self.service.observation = ({'id': 'o', 'state': state}, time.monotonic(), None)
        self.service._preflight = Mock(return_value=(self.start, [self.knots]))
        pos = self.scene_state.joint_state.name.index('right_s0')
        self.guard._valid.side_effect = lambda s: NS(valid=not math.isclose(s.joint_state.position[pos], .04), contacts=[])
        with patch('cix_harness.primitive_service.plan', return_value=[{'kind':'move','motion_phase':'transit'}]):
            result = self.service.prepare('o', 'move_to', target_base_m=[.60,.20,.0])
        self.assertEqual(result['failure_code'], 'moveit_collision')
        self.assertFalse(result['applied'])
        self.assertIsNone(self.service.prepared)
        self.r.move_segment.assert_not_called()
        self.r.grip.command_position.assert_not_called()

    def test_unavailable_at_execute_never_sends_planned_move(self):
        self.service.prepared = ('p', time.monotonic(), self.start,
                                 [{'kind': 'move', 'motion_phase': 'transit'}], [self.knots], 1, 'move_to', False)
        self.guard._scene.side_effect = RuntimeError('state service disconnected')
        receipt=self.service.execute('p')
        self.assertIn('service disconnected',receipt['message'])
        self.assertIs(receipt['applied'],False);self.assertFalse(receipt['stop_attempted'])
        self.assertEqual(receipt['failed_stage']['command_state'],'not_dispatched')
        self.r.move_segment.assert_not_called()
        self.r.grip.command_position.assert_not_called()
        self.assertIsNone(self.guard.execution_reference)
        self.r.limb.set_joint_positions.assert_not_called()
        self.r.grip.stop.assert_not_called()
        self.assertEqual(self.r.motion_remaining, 20)

    def test_scene_change_and_stale_or_missing_peer_fail_closed(self):
        self.guard._scene.side_effect = [(self.scene_state, self.scene_report), (self.scene_state, {})]
        with self.assertRaisesRegex(RuntimeError, 'scene geometry'):
            self.guard.check(self.start, [(0, 'transit', self.knots)])
        for sample in [(time.monotonic()-2, self.full), (time.monotonic(), self.start)]:
            self.r.full_body_joint_sample.side_effect = None
            self.r.full_body_joint_sample.return_value = sample
            with self.assertRaisesRegex(RuntimeError, 'both arms and head'):
                self.guard.peer()

    def test_corrupt_configuration_does_not_fall_back(self):
        for p in ({'collision_backend':'moveitt'}, {'moveit_required_world_objects':['table']},
                  {'collision_backend':'moveit','moveit_required_world_objects':['','table']}):
            with self.assertRaises(ValueError):
                options(p)

    def test_survey_box_requires_size_source_and_rigid_frame(self):
        box={'id':'mounted_board','frame':'right_gripper','xyz':[0,.01,0],'xyzw':[0,0,0,1],
             'size_m':[.11,.11,.008],'source':'on-site measurement'}
        self.assertEqual(options({'collision_backend':'moveit','moveit_scene_boxes':[box]}),('moveit',[]))
        for change in ({'source':''},{'frame':'camera'},{'size_m':[.1,.1,0]},{'xyzw':[0,0,0,2]}):
            with self.assertRaises(ValueError):
                options({'collision_backend':'moveit','moveit_scene_boxes':[dict(box,**change)]})
        with self.assertRaises(ValueError):
            options({'collision_backend':'moveit','moveit_scene_boxes':[box,box]})
        with self.assertRaises(ValueError):
            options({'moveit_scene_boxes':[box]})

    def test_request_local_table_and_peer_board_enter_whole_robot_state(self):
        from cix_harness.moveit_guard import MoveItGuard
        self.guard.ros=NS(Time=lambda sec:sec)
        state=NS(joint_state=NS(name=list(self.full),position=list(self.full.values())),attached_collision_objects=[])
        live=NS(robot_model_name='baxter',robot_state=state,world=NS(collision_objects=[]),allowed_collision_matrix=object())
        self.guard.scene_service=Mock(return_value=NS(scene=live))
        self.guard.required_world=['surface']
        self.r.cfg['planning']['moveit_scene_boxes']=[
            dict(id=k,frame=f,source='measurement',size_m=[.1,.1,.01])
            for k,f in [('surface','base'),('installed_board','right_gripper')]]
        c=NS(SCENE_SETTINGS=1,ROBOT_STATE=2,ROBOT_STATE_ATTACHED_OBJECTS=4,WORLD_OBJECT_GEOMETRY=8,
             ALLOWED_COLLISION_MATRIX=16)
        def entry(box):
            return NS(object=NS(header=NS(stamp=0),id=box['id']),link_name=box['frame'],touch_links=[box['frame']])
        with patch.dict('sys.modules',{'moveit_msgs.msg':NS(PlanningSceneComponents=c),
                                       'moveit_msgs.srv':NS(GetPlanningSceneRequest=lambda:NS(components=NS()))}), \
                patch('cix_harness.moveit_guard.scene_box',side_effect=entry), \
                patch('cix_harness.moveit_guard.message_digest',return_value='digest'):
            checked,receipt=MoveItGuard._scene(self.guard)
        self.assertEqual([(a.object.id,a.link_name) for a in checked.attached_collision_objects],
                         [('surface','base'),('installed_board','right_gripper')])
        self.assertEqual(receipt['world_objects']['surface']['scope'],'request_fixed_to_base')
        self.assertEqual(receipt['attached_objects']['installed_board']['link'],'right_gripper')
