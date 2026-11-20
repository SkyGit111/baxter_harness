import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from cix_harness.moveit_probe import assess_ros, joint_names, main, model_inventory, trajectory_summary


URDF = '<robot name="baxter"><link name="torso"><collision><geometry><cylinder radius="0.1" length="0.5"/></geometry></collision></link></robot>'
SRDF = '<robot name="baxter"><group name="right_arm"/><disable_collisions link1="fixed1" link2="fixed2" reason="Adjacent"/></robot>'


class MoveItAssessmentTests(unittest.TestCase):
    def test_offline_entry_uses_right_tcp_and_model_semantics(self):
        with TemporaryDirectory() as d:
            urdf, srdf, config, report = (Path(d)/n for n in ('r.xml', 's.xml', 'site.json', 'report.json'))
            urdf.write_text(URDF)
            srdf.write_text(SRDF)
            config.write_text(json.dumps({'arms': {'right': {'primitives': {'tcp_offset_tool_m': [0,0,0.1]}}}}))
            self.assertEqual(main(['--urdf', str(urdf), '--srdf', str(srdf), '--arm', 'right',
                                   '--tcp-base-m', '0.5', '0', '0.2', '--eef-xyzw', '1', '0', '0', '0',
                                   '--config', str(config), '--output', str(report)]), 0)
            result = json.loads(report.read_text())
            self.assertEqual(result['model_inventory']['shape_counts'], {'cylinder': 1})
            self.assertEqual(result['model_inventory']['configured_groups'], ['right_arm'])
            self.assertEqual(len(result['model_inventory']['disabled_collision_pairs']), 1)
            self.assertFalse(result['applied'])
            self.assertFalse(result['complete_task_checked'])
            goal = result['request']['goal_constraints'][0]['position_constraints'][0]
            self.assertEqual((goal['header']['frame_id'], goal['link_name']), ('base', 'right_gripper'))
            self.assertAlmostEqual(goal['constraint_region']['primitive_poses'][0]['position']['z'], 0.3)
            self.assertNotIn('path_constraints', result['request'])

    def test_unmatched_semantic_model_rejected(self):
        with self.assertRaisesRegex(ValueError, 'model names'):
            model_inventory(URDF, SRDF.replace('baxter', 'panda'))

    def test_invalid_scene_state_records_collision_pair_before_planning(self):
        names = sorted(joint_names('right') | joint_names('left'))
        joints = NS(name=names, position=[0.]*14, header=NS(stamp=NS(to_sec=lambda: 1.)))
        state = NS(joint_state=joints, attached_collision_objects=[])
        scene = NS(name='bench', robot_model_name='baxter', robot_state=state,
                   world=NS(collision_objects=[]), allowed_collision_matrix=NS())
        ros = Mock()
        ros.get_param.side_effect = [URDF, SRDF]
        ros.Time.now.return_value.to_sec.return_value = 1.
        contact = NS(contact_body_1='collision_head_link_1', contact_body_2='right_forearm', depth=.01)
        services = {'/get_planning_scene': lambda req: NS(scene=scene),
                    '/check_state_validity': lambda req: NS(valid=False, contacts=[contact])}
        ros.ServiceProxy.side_effect = lambda name, _: services[name]
        components = NS(SCENE_SETTINGS=1, ROBOT_STATE=2, ROBOT_STATE_ATTACHED_OBJECTS=4,
                        WORLD_OBJECT_GEOMETRY=16, ALLOWED_COLLISION_MATRIX=128)
        srv = NS(GetPlanningScene=object, GetMotionPlan=object, GetStateValidity=object,
                 GetPlanningSceneRequest=lambda: NS(components=NS()), GetMotionPlanRequest=NS,
                 GetStateValidityRequest=NS)
        genpy = NS(message=NS(strify_message=lambda obj: ''))
        with patch.dict('sys.modules', {'rospy': ros, 'genpy': genpy, 'genpy.message': genpy.message,
                                       'moveit_msgs': NS(), 'moveit_msgs.srv': srv,
                                       'moveit_msgs.msg': NS(PlanningSceneComponents=components)}):
            report = assess_ros({'group_name': 'right_arm'}, 'right')
        self.assertEqual(report['status'], 'invalid_start_state')
        self.assertEqual(report['start_contacts'][0]['bodies'], ['collision_head_link_1', 'right_forearm'])
        self.assertEqual([call.args[0] for call in ros.ServiceProxy.call_args_list], list(services))

    def test_requires_actual_single_arm_path_on_success(self):
        names = sorted(joint_names('right'))
        def point(q, t):
            return NS(positions=q, time_from_start=NS(to_sec=lambda: t))
        path = NS(joint_names=names, points=[point([0]*7, 0), point([.1]*7, 2)])
        response = NS(error_code=NS(val=1), planning_time=.1, trajectory=NS(joint_trajectory=path))
        self.assertEqual(trajectory_summary(response, 'right')['trajectory_duration_s'], 2)
        self.assertEqual(trajectory_summary(response, 'right')['joint_travel_rad'][names[0]], .1)
        with self.assertRaisesRegex(ValueError, 'group'):
            trajectory_summary(response, 'left')
        path.points.reverse()
        with self.assertRaisesRegex(ValueError, 'timing'):
            trajectory_summary(response, 'right')
        path.points.clear()
        with self.assertRaisesRegex(ValueError, 'no points'):
            trajectory_summary(response, 'right')
        response.error_code.val = -1
        self.assertEqual(trajectory_summary(response, 'right')['status'], 'planning_failed')
