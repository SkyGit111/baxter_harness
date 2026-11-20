import contextlib
import copy
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from cix_harness import prepare_site, table_survey


class TableSurveyTests(unittest.TestCase):
    def setUp(self):
        self.extrinsics = {'from_frame':'camera_color_optical_frame','to_frame':'base','units':'m',
                           'translation_m':[.5,0,1.],'xyzw':[1,0,0,0],'source':'synthetic transformed camera'}
        self.survey = {'units':'m','source':'synthetic tilted tabletop','boundary': {
            'frame':'camera_color_optical_frame',
            'points_m':[[-.2,-.3,1.17],[.2,-.3,1.16],[.2,.3,1.16],[-.2,.3,1.17]]},
            'check_points':{'frame':'base','points_m':[[.5,0,-.165]]}}

    def fit(self):
        return table_survey.build_geometry(self.extrinsics, self.survey, max_height_error_m=.005)

    def test_handoff_from_camera_measurements_to_uncommissioned_profiles(self):
        with TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            folder = Path(folder)
            ext, inputs, output = (folder/name for name in ('ext.json', 'survey.json', 'geometry.json'))
            ext.write_text(json.dumps(self.extrinsics))
            inputs.write_text(json.dumps(self.survey))
            table_survey.main(['--extrinsics', str(ext), '--survey', str(inputs), '--output', str(output),
                               '--max-height-error-m', '.005'])
            geom = json.loads(output.read_text())
            self.assertFalse(geom['commissioned'])
            self.assertEqual(geom['inputs']['extrinsics']['file'], 'ext.json')
            self.assertLess(geom['independent_check']['max_abs_m'], 1e-9)
            template = Path(__file__).resolve().parents[1]/'config/baxter-primitives.example.json'
            prepare_site.main(['--template', str(template), '--geometry', str(output),'--output-dir', str(folder)])
            for arm in ('left','right'):
                profile = json.loads((folder/(arm+'-fulltable.local.json')).read_text())
                self.assertFalse(profile['commissioned'])
                self.assertFalse(profile['primitives']['calibrated'])
                self.assertEqual(profile['primitives']['table_polygon_base_xy'][0], [.3,.3])

    def test_bad_frame_unit_quaternion_and_perimeter_rejected(self):
        for key, value in [('from_frame','base'),('to_frame','camera_color_optical_frame'),('units','mm'),
                           ('xyzw',[0,0,0,2])]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                table_survey.build_geometry(dict(self.extrinsics, **{key:value}), self.survey, max_height_error_m=.005)
        for points in ([[0,0,1],[.1,0,1],[.2,0,1]],
                       [[0,0,1],[1,1,1],[0,1,1],[1,0,1]],
                       [[0,0,1],[1,0,1],[.5,.2,1],[1,1,1],[0,1,1]]):
            with self.subTest(points=points), self.assertRaises(ValueError):
                survey = copy.deepcopy(self.survey)
                survey['boundary']['points_m'] = points
                table_survey.build_geometry(self.extrinsics, survey, max_height_error_m=.005)
        self.survey['boundary']['frame'] = 'camera_link'
        with self.assertRaises(ValueError):
            self.fit()

    def test_bowed_perimeter_and_independent_disagreement_rejected(self):
        self.survey['check_points']['points_m'][0][2] = -.22
        with self.assertRaisesRegex(ValueError, 'independent check vertical residual'):
            self.fit()
        del self.survey['check_points']
        self.assertIsNone(self.fit()['independent_check']['rms_m'])
        self.survey['boundary']['points_m'][0][2] += .1
        with self.assertRaisesRegex(ValueError, 'boundary vertical residual'):
            self.fit()

    def test_independent_3d_correspondence_catches_in_plane_translation(self):
        self.survey['reference_pairs'] = [{'color_optical_m': [0.,0.,1.165], 'measured_base_m': [.5,0.,-.165]}]
        args = {'max_height_error_m':.005, 'max_reference_error_m':.008}
        result = table_survey.build_geometry(self.extrinsics, self.survey, **args)
        self.assertLess(result['reference_errors_m'][0], 1e-9)
        self.extrinsics['translation_m'][1] += .13
        with self.assertRaisesRegex(ValueError, '3-D reference error'):
            table_survey.build_geometry(self.extrinsics, self.survey, **args)
