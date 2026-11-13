"""Projection and planner context tests without ROS, camera SDK or API access."""
import copy
import math
import json
from pathlib import Path
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from cix_harness.task_plan import VisualTaskPlanner
from cix_harness.workspace import workspace_metadata
from cix_harness.primitive_service import PrimitiveService


def pinhole(intrinsics, point):
    return [intrinsics.fx * point[0] / point[2] + intrinsics.ppx,
            intrinsics.fy * point[1] / point[2] + intrinsics.ppy]


def profile():
    return {'tcp_workspace_min_m': [-.2, -.1, 0], 'tcp_workspace_max_m': [.2, .1, .5],
            'table_z_m': 1., 'camera_to_base_xyzw': [0, 0, 0, 1],
            'camera_to_base_translation_m': [0, 0, 0], 'calibrated': False,
            'notes': 'Synthetic geometry; uncertainty remains.'}


class WorkspaceMetadataTests(unittest.TestCase):
    def setUp(self):
        self.intrinsics = NS(width=101, height=101, fx=100., fy=100., ppx=50., ppy=50.)

    def test_known_table_projection_preserves_corner_order_and_profile(self):
        p = profile(); before = copy.deepcopy(p)
        result = workspace_metadata(p, self.intrinsics, pinhole)
        self.assertEqual(result['table_reference']['polygon_uv'], [[.3, .4], [.7, .4], [.7, .6], [.3, .6]])
        self.assertEqual(result['table_reference']['status'], 'in_image')
        self.assertEqual(result['tcp_min_m'], p['tcp_workspace_min_m'])
        self.assertFalse(result['profile_calibrated'])
        self.assertEqual(p, before)

    def test_inverse_camera_rotation_translation_and_distortion_callback(self):
        p = profile(); p['camera_to_base_xyzw'] = [0, 0, math.sqrt(.5), math.sqrt(.5)]
        p['camera_to_base_translation_m'] = [.1, .2, 0]
        project = Mock(side_effect=pinhole)
        result = workspace_metadata(p, self.intrinsics, project)
        # Base [-.2,-.1,1] minus translation => [-.3,-.3,1]; inverse Z90 => [-.3,.3,1].
        first = project.call_args_list[0].args[1]
        for actual, expected in zip(first, [-.3, .3, 1.]):
            self.assertAlmostEqual(actual, expected)
        self.assertEqual(result['table_reference']['polygon_uv'][0], [.2, .8])
        self.assertEqual(project.call_count, 4)

    def test_outside_image_corners_are_not_clamped(self):
        p = profile(); p['tcp_workspace_min_m'][0] = -2
        result = workspace_metadata(p, self.intrinsics, pinhole)
        self.assertLess(result['table_reference']['polygon_uv'][0][0], 0)
        self.assertEqual(result['table_reference']['status'], 'extends_outside_image')

    def test_reference_behind_camera_never_yields_a_plausible_polygon(self):
        p = profile(); p['camera_to_base_translation_m'][2] = 2
        project = Mock()
        result = workspace_metadata(p, self.intrinsics, project)
        self.assertIsNone(result['table_reference']['polygon_uv'])
        self.assertEqual(result['table_reference']['status'], 'not_in_front_of_camera')
        project.assert_not_called()

    def test_current_geometry_reaches_both_plan_and_check_without_mutating_context(self):
        planner = VisualTaskPlanner(1, 'offline'); planner.request = Mock(return_value={})
        metadata = workspace_metadata(profile(), self.intrinsics, pinhole)
        observation = {'workspace': metadata}
        original = {'route': 'hybrid', 'workspace': {'obsolete': True}}
        for method in (planner.plan, planner.check):
            method(observation, original)
            arguments = planner.request.call_args.args
            self.assertEqual(arguments[4]['workspace'], metadata)
            self.assertIn('parallax', arguments[1])
            self.assertIn('not invent free space or hidden pixels', arguments[1])
            self.assertEqual(original['workspace'], {'obsolete': True})
            method({}, original)
            self.assertNotIn('workspace', planner.request.call_args.args[4])

    def test_notes_are_bounded_and_nonfinite_projection_is_rejected(self):
        p = profile(); p['notes'] = 'x' * 2000
        result = workspace_metadata(p, self.intrinsics, pinhole)
        self.assertEqual(len(result['calibration_note']), 300)
        with self.assertRaises(ValueError):
            workspace_metadata(p, self.intrinsics, lambda _i, _p: [float('nan'), 2])

    def test_observation_publishes_current_geometry_without_actuating(self):
        p = json.loads(Path('config/baxter-primitives.example.json').read_text())['primitives']
        p['camera_to_base_translation_m'] = [0, 0, -1]
        robot = Mock()
        robot.cfg = {'primitives': p}; robot.motion_remaining = 20
        robot.camera.intrinsics = self.intrinsics
        robot.camera.last_depth = NS(copy=lambda: 'synthetic-depth-copy')
        robot.observe.return_value = {'id': 'fresh', 'state': {'eef_m': [.6, .2, .1], 'eef_xyzw': [1, 0, 0, 0]}}
        service = PrimitiveService(robot)
        with patch.dict('sys.modules', {'pyrealsense2': NS(rs2_project_point_to_pixel=pinhole)}):
            observation = service.observe()
        self.assertEqual(observation['workspace']['tcp_min_m'], p['tcp_workspace_min_m'])
        self.assertEqual(observation['workspace']['table_z_m'], p['table_z_m'])
        self.assertIsNotNone(observation['workspace']['table_reference']['polygon_uv'])
        self.assertEqual(service.observation[0]['id'], 'fresh')
        robot.move_segment.assert_not_called()
        robot.grip.command_position.assert_not_called()


if __name__ == '__main__':
    unittest.main()
