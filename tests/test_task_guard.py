import unittest
import math
from cix_harness.pi05_contract import multiply
from cix_harness.task_guard import check_contact_pose
from cix_harness.task_monitor import TaskMonitor


class TaskGuardTest(unittest.TestCase):
    def setUp(self):
        self.guard={'source':'operator_confirmed_fingertips_at_table','reference_eef_m':[.8,.1,-.1673],
                    'reference_eef_xyzw':[0,1,0,0],'clearance_m':.008,'radius_m':.25,
                    'max_tilt_rad':.08,'tool_extent_bound_m':.2}

    def test_touch_reference_does_not_use_nominal_table_as_endpoint_height(self):
        check_contact_pose(self.guard,[.8,.1,-.15],[0,1,0,0])
        with self.assertRaisesRegex(ValueError,'clearance'):
            check_contact_pose(self.guard,[.8,.1,-.17],[0,1,0,0])
        with self.assertRaisesRegex(ValueError,'region'):
            check_contact_pose(self.guard,[1.1,.1,-.15],[0,1,0,0])
        check_contact_pose(self.guard,[1.1,.1,.2],[0,1,0,0])
        with self.assertRaisesRegex(ValueError,'clearance'):
            check_contact_pose(self.guard,[.8,.1,0],[0,0,0,1])
        # Orientation alone no longer trips a fixed angular threshold.
        check_contact_pose(self.guard,[.8,.1,.5],[0,0,0,1])

    def test_grasp_lift_release_evidence_is_required_for_stack(self):
        monitor=TaskMonitor('stack')
        monitor.observe({'eef_m':[.8,.1,-.1],'gripping':False,'gripper_percent':100})
        self.assertFalse(monitor.can_complete())
        monitor.observe({'eef_m':[.8,.1,-.1],'gripping':True,'gripper_percent':40})
        monitor.observe({'eef_m':[.8,.1,-.09],'gripping':True,'gripper_percent':40})
        self.assertFalse(monitor.can_complete())
        monitor.observe({'eef_m':[.8,.1,-.06],'gripping':True,'gripper_percent':40})
        self.assertTrue(monitor.held_and_lifted)
        self.assertFalse(monitor.can_complete())
        monitor.applied({'endpoint_after_m':[.8,.1,-.1],'gripping':False,'gripper_percent':100},[0,0,0,0,0,0,1])
        self.assertTrue(monitor.can_complete())

    def test_local_yaw_of_tilted_tool_changes_clearance_bound(self):
        tilted=multiply([0,1,0,0],[math.sin(.02),0,0,math.cos(.02)])
        guard={**self.guard,'reference_eef_xyzw':tilted}
        local_yaw=multiply(tilted,[0,0,math.sin(.1),math.cos(.1)])
        bound=check_contact_pose(guard,[.8,.1,-.1],local_yaw)
        self.assertAlmostEqual(bound['tool_tilt_rad'],0,places=6)
        self.assertGreater(bound['orientation_margin_m'],.001)

    def test_planning_buffer_is_not_the_measured_contact_stop_line(self):
        with self.assertRaisesRegex(ValueError,'clearance'):
            check_contact_pose(self.guard,[.8,.1,-.160],[0,1,0,0])
        check_contact_pose(self.guard,[.8,.1,-.160],[0,1,0,0],enforce_planning_margin=False)
        with self.assertRaisesRegex(ValueError,'clearance'):
            check_contact_pose(self.guard,[.8,.1,-.170],[0,1,0,0],enforce_planning_margin=False)

    def test_dropped_object_cannot_be_reported_as_placement(self):
        monitor=TaskMonitor('stack')
        monitor.observe({'eef_m':[0,0,0],'gripping':True,'gripper_percent':40})
        monitor.observe({'eef_m':[0,0,.04],'gripping':True,'gripper_percent':40})
        monitor.observe({'eef_m':[0,0,.04],'gripping':False,'gripper_percent':100})
        self.assertFalse(monitor.can_complete())
        with self.assertRaisesRegex(RuntimeError,'lost'):
            monitor.require_progress()

    def test_empty_regrasp_does_not_inherit_prior_lift(self):
        monitor=TaskMonitor('pick')
        monitor.observe({'eef_m':[0,0,0],'gripping':True})
        monitor.observe({'eef_m':[0,0,.04],'gripping':True})
        self.assertTrue(monitor.can_complete())
        monitor.applied({'endpoint_after_m':[0,0,.04],'gripping':False,'gripper_percent':100},[0]*6+[1])
        monitor.observe({'eef_m':[0,0,.04],'gripping':True})
        self.assertFalse(monitor.can_complete())
