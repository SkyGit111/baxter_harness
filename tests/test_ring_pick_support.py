import unittest

from cix_harness.ring_peg import RingGeometryError, pick_geometry
from tests.test_ring_peg import example


class SurveyedFingertipContact(unittest.TestCase):
    def setUp(self):
        self.profile, self.scene, self.state, self.ring, _ = example()
        settings = self.profile['ring_peg']
        settings['tcp_table_error_m'] = .0025
        settings['local_pick_support'] = {
            'ring_id': self.ring['id'], 'frame_ids': [2, 3, 4, 5, 6],
            'jaw_axis_base_xy': [1., 0.], 'swept_table_upper_z_m': .034,
            'ring_top_z_m': .039, 'ring_top_upper_z_m': .041
        }
        self.ring.update(local_table_z_m=.030, local_table_upper_z_m=.051, local_table_depth_count=500,
                         local_support_reference_verified=True, local_support_reference_evidence={
                             'verified': True, 'survey_frame_ids': [2, 3, 4, 5, 6], 'current_surface_shift_m': 0.,
                             'jaw_axis_base_xy': [1., 0.]})

    def contact(self, quat=None):
        return pick_geometry(self.profile, self.scene, self.ring, quat or self.state['eef_xyzw'])

    def test_uses_unchanged_directional_footprint_and_accounts_for_lowest_actual_finger(self):
        contact, output = self.contact()
        self.assertAlmostEqual(contact[2], .038)
        self.assertEqual(output['local_table_upper_z_m'], .034)
        self.assertEqual(output['support_survey']['empty_grasp_possible'], False)
        self.ring['local_support_reference_evidence']['current_surface_shift_m'] = -.002
        self.assertAlmostEqual(self.contact()[0][2], .038)
        self.assertTrue(self.contact()[1]['support_survey']['empty_grasp_possible'])
        self.ring['local_support_reference_evidence']['current_surface_shift_m'] = .002
        self.assertAlmostEqual(self.contact()[0][2], .040)

    def test_changed_site_direction_or_source_cannot_reuse_the_lower_floor(self):
        with self.assertRaises(RingGeometryError):
            self.contact([0., 0., .707106781, .707106781])
        self.ring['local_support_reference_evidence']['survey_frame_ids'][0] = 1
        with self.assertRaises(RingGeometryError):
            self.contact()
        self.ring['local_support_reference_verified'] = False
        with self.assertRaises(RingGeometryError):
            self.contact()

    def test_even_highest_measured_ring_top_below_safe_fingers(self):
        self.profile['ring_peg']['local_pick_support']['swept_table_upper_z_m'] = .041
        with self.assertRaisesRegex(RingGeometryError, 'No fingertip contact depth'):
            self.contact()

    def test_contact_upper_band_may_yield_a_clean_miss_without_lowering_the_table_floor(self):
        self.profile['ring_peg']['local_pick_support']['swept_table_upper_z_m'] = .0395
        contact, evidence = self.contact()
        self.assertGreater(contact[2], .041)
        self.assertGreaterEqual(evidence['support_survey']['highest_fingertip_at_lowest_endpoint_m'], .040)
        self.assertTrue(evidence['support_survey']['empty_grasp_possible'])
