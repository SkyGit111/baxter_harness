import math
import unittest
from cix_harness.segments import compose_prefix,euler_quaternion
from cix_harness.pi05_contract import rotate


class SegmentTest(unittest.TestCase):
    def test_small_predictions_form_one_centimeter_scale_segment(self):
        action,used,reason,length=compose_prefix([[.006,0,0,0,0,0,1]]*16,True)
        self.assertEqual(used,16)
        self.assertAlmostEqual(action[0],.096)
        self.assertAlmostEqual(length,.096)
        self.assertEqual(reason,'window_complete')

    def test_closure_ends_segment_at_the_predicted_transition(self):
        action,used,reason,_=compose_prefix([[.01,0,0,0,0,0,1],
                                           [.01,0,0,0,0,0,0],
                                           [0,0,.05,0,0,0,0]],True)
        self.assertEqual(used,2)
        self.assertEqual(action,[.02,0,0,0,0,0,0])
        self.assertEqual(reason,'gripper_transition')

    def test_reversal_is_not_shortcut_and_invalid_suffix_rejects_whole_request(self):
        a,n,r,_=compose_prefix([[.01,0,0,0,0,0,1],[-.02,0,0,0,0,0,1]],True)
        self.assertEqual(n,1)
        self.assertEqual(r,'direction_change')
        with self.assertRaises(ValueError):
            compose_prefix([[.01,0,0,0,0,0,0],[float('nan')]*7],True)

    def test_path_budget_and_rotation_composition(self):
        a,n,r,length=compose_prefix([[.04,0,0,0,0,.1,1]]*4,True)
        self.assertEqual(n,3)
        self.assertAlmostEqual(length,.12)
        self.assertEqual(r,'path_budget')
        v=rotate(euler_quaternion(a[3:6]),[1,0,0])
        self.assertAlmostEqual(v[0],math.cos(.3))
