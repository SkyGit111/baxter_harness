import json
import math
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from cix_harness.moveit_guard import options, segment_samples
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
        with self.assertRaisesRegex(RuntimeError, 'service disconnected'):
            self.service.execute('p')
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
