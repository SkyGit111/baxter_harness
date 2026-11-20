import contextlib
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from cix_harness import handeye, handeye_volume as volume


class SpatialCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.bounds=dict(frame='base',units='m',lower=[.42,-.42,.08],upper=[.72,-.10,.28])
        self.Y=np.eye(4);self.Y[:3,3]=[.08,.015,.04]
        self.X=np.eye(4);self.X[:3,3]=[.55,-.26,-.8]
        self.reference={'base_T_effector':handeye.packed(np.eye(4)), 'effector_T_tag':handeye.packed(self.Y)}
        self.plan=volume.generate_plan(self.bounds,self.reference)
        self.manifest={'intrinsics':{'width':640,'height':480}}
        self.fit=[];self.hold=[]
        for p in self.plan['poses']:
            E=handeye.transform(p['base_T_effector'])
            C=np.linalg.solve(self.X,E@self.Y)
            u,v=500*C[:2,3]/C[2,3]+[320,240]
            corners=[[u+a,v+b] for a,b in ((-15,-15),(15,-15),(15,15),(-15,15))]
            sample={'drive_pose_id':p['id'],'base_T_effector':E.tolist(),'camera_T_tag':C.tolist(),'corners_px':corners}
            (self.fit if p['split']=='train' else self.hold).append(sample)

    def audit(self):
        return volume.audit_volume(self.plan,self.fit,self.hold,self.X,self.manifest)

    def test_requested_locations_refer_to_visible_marker_and_fill_independent_cells(self):
        self.assertEqual((len(self.fit),len(self.hold)),(24,6))
        stats=self.audit()
        self.assertTrue(stats['passed'],stats['failures'])
        for p in self.plan['poses']:
            actual=handeye.transform(p['base_T_effector'])@self.Y
            np.testing.assert_allclose(p['tag_center_base_m'],actual[:3,3])
        self.assertEqual(len(stats['groups']['train']['cell_counts']),12)
        self.assertEqual(len(stats['groups']['holdout']['cell_counts']),6)
        self.assertGreater(stats['groups']['holdout']['tag_optical']['pca_rms_mm'][2],12.)
        self.assertLess(stats['groups']['train']['center_uv_min'][0],stats['groups']['train']['center_uv_max'][0])

    def test_many_wrist_poses_cannot_mask_a_stationary_or_planar_tag(self):
        for sample in self.fit+self.hold:
            sample['camera_T_tag'][2][3]=.99
        stats=self.audit()
        self.assertFalse(stats['passed'])
        self.assertGreater(stats['groups']['train']['wrist_base']['span_mm'][2],100.)
        self.assertAlmostEqual(stats['groups']['train']['tag_optical']['pca_rms_mm'][2],0.)

    def test_relabeling_missing_id_and_tiny_planned_region(self):
        self.fit[0],self.hold[0]=self.hold[0],self.fit[0]
        self.assertFalse(self.audit()['passed'])
        tiny=dict(self.bounds,upper=[.45,-.4,.10])
        with self.assertRaisesRegex(ValueError,'too small'):volume.generate_plan(tiny,self.reference)
        bad=deepcopy(self.plan);bad['poses'][1]['id']=bad['poses'][0]['id']
        with self.assertRaisesRegex(ValueError,'Duplicate'):volume.validate_plan(bad)

    def test_rigid_fit_only_exports_the_matching_spatial_capture(self):
        with TemporaryDirectory() as path, contextlib.redirect_stdout(io.StringIO()):
            d=Path(path)
            for n,data in [('plan',self.plan),('fit',self.fit),('holdout',self.hold),('manifest',self.manifest)]:
                (d/(n+'.json')).write_text(json.dumps(data))
            args=['--fit',str(d/'fit.json'),'--holdout',str(d/'holdout.json'),'--volume-plan',str(d/'plan.json'),
                  '--max-mm','5','--max-deg','2','--optical-frame','camera_color_optical_frame',
                  '--extrinsics-output',str(d/'camera.json')]
            self.assertEqual(handeye.main(args+['--output',str(d/'bad.json')]),2)
            self.assertFalse((d/'camera.json').exists())
            self.assertTrue(json.loads((d/'bad.json').read_text())['consistency_passed'])
            self.manifest['volume_plan_sha256']=hashlib.sha256((d/'plan.json').read_bytes()).hexdigest()
            (d/'manifest.json').write_text(json.dumps(self.manifest))
            ids={p['id']:p for p in self.plan['poses']}
            for n,rows in [('fit',self.fit),('holdout',self.hold)]:
                for sample in rows:
                    pose=ids[sample['drive_pose_id']]
                    sample.update(volume_plan_sha256=self.manifest['volume_plan_sha256'],
                                  volume_split=pose['split'],volume_cell=pose['cell'])
                (d/(n+'.json')).write_text(json.dumps(rows))
            self.assertEqual(handeye.main(args+['--output',str(d/'ok.json')]),0)
            self.assertTrue((d/'camera.json').exists())

    def test_stationary_estimator_uses_ruler_corners_and_same_centres_for_spatial_audit(self):
        model=np.array([[-.05,-.05,0],[.05,-.05,0],[.05,.05,0],[-.05,.05,0]])
        truth={}
        for serial,sample in enumerate(self.fit+self.hold):
            C=np.array(sample['camera_T_tag']);truth[sample['drive_pose_id']]=(self.X@C)[:3,3]
            frames=[]
            for i in range(5):
                observed=C.copy()
                if i==4:
                    a=np.radians(1.4)*(-1)**serial
                    observed[:3,:3]=C[:3,:3]@np.array([[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]])
                    observed[0,3]+=.003*(-1)**serial
                stamp=serial+1.+i*.14
                frames.append({**sample,'camera_T_tag':observed.tolist(),'corners_optical_m':(model@observed[:3,:3].T+observed[:3,3]).tolist(),
                               'color_frame':i,'host_time_s':stamp,'endpoint_ros_stamp_s':stamp,'joint_ros_stamp_s':stamp,
                               'color_stamp_ms':stamp*1000,'depth_stamp_ms':stamp*1000})
            sample.update(camera_T_tag=frames[-1]['camera_T_tag'],stationary_frames=frames,color_frame=4,
                          ruler_reference={'width_m':.1,'height_m':.1})
        with TemporaryDirectory() as folder, contextlib.redirect_stdout(io.StringIO()):
            d=Path(folder);(d/'plan.json').write_text(json.dumps(self.plan))
            sha=hashlib.sha256((d/'plan.json').read_bytes()).hexdigest()
            by_id={p['id']:p for p in self.plan['poses']}
            for split,rows in [('train',self.fit),('holdout',self.hold)]:
                for sample in rows:
                    sample.update(volume_plan_sha256=sha,volume_split=split,volume_cell=by_id[sample['drive_pose_id']]['cell'])
                (d/(split+'.json')).write_text(json.dumps(rows))
            (d/'manifest.json').write_text(json.dumps({**self.manifest,'volume_plan_sha256':sha}))
            args=['--fit',str(d/'train.json'),'--holdout',str(d/'holdout.json'),'--volume-plan',str(d/'plan.json'),
                  '--pose-estimator','stationary-window','--max-mm','1','--max-deg','.1','--output',str(d/'report.json')]
            self.assertEqual(handeye.main(args),0)
            result=json.loads((d/'report.json').read_text())
            self.assertEqual(result['pose_estimator'],'stationary-window')
            self.assertFalse(result['last_frame_diagnostic']['consistency_passed'])
            self.assertLess(result['holdout']['rms_deg'],.001)
            self.assertTrue(all(x['frame_count']==5 for x in result['stationary_windows']['fit']))
            for row in result['workspace_coverage']['groups']['holdout']['samples']:
                np.testing.assert_allclose(truth[row['id']],row['tag_base_m'],atol=1e-9)

    def test_window_estimator_rejects_unavailable_evidence(self):
        from cix_harness.handeye_window import estimate_rows
        with self.assertRaisesRegex(ValueError,'Missing synchronized'):
            estimate_rows(self.fit,'stationary-window')
