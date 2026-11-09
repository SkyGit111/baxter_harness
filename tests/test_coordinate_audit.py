import unittest

from cix_harness.coordinate_audit import motion_report, state_report
from cix_harness.pi05 import Pi05


class CoordinateAuditTest(unittest.TestCase):
    def test_distinguishes_reversed_tracking_from_wrong_policy_direction(self):
        receipt = {'obs_id': 'test', 'endpoint_before_m': [1, 2, 3],
                   'endpoint_target_m': [1, 2.1, 3], 'endpoint_after_m': [1, 2.09, 3]}
        self.assertAlmostEqual(motion_report(receipt)['direction_error_deg'], 0)
        receipt['endpoint_after_m'] = [1, 1.9, 3]
        self.assertAlmostEqual(motion_report(receipt)['direction_error_deg'], 180)
        receipt['endpoint_after_m'] = receipt['endpoint_before_m']
        self.assertIsNone(motion_report(receipt)['direction_error_deg'])

    def test_token_saturation_preserves_upstream_minus_one_bin(self):
        stats = {'mean': [0]*8, 'std': [1-1e-8]*8, 'q01': [-1]*8, 'q99': [1]*8,
                 'min': [-3]*8, 'max': [3]*8}
        report = state_report([-2, -1, 0, 1, 2, 3, 4, 5], stats)
        self.assertEqual(report['state_tokens8'], [-1, 0, 128, 255, 255, 255, 255, 255])
        self.assertEqual(report['outside_training_min_max'], ['finger_left', 'finger_right'])

    def test_seeded_diagnostics_cannot_run_with_commissioned_profile(self):
        service = Pi05.__new__(Pi05)
        service.profile = {'commissioned': True}
        with self.assertRaisesRegex(ValueError, 'uncommissioned'):
            service.predict({}, {'diagnostic_seed': 42})
        for invalid in [True, -1, 2**32, 1.5]:
            with self.assertRaisesRegex(ValueError, 'integer'):
                service.predict({}, {'diagnostic_seed': invalid})
