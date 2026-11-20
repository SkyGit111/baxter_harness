from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from cix_harness.handeye_frozen import evaluate


class FrozenCaptureTest(unittest.TestCase):
    def setUp(self):
        tmp=TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root=Path(tmp.name)
        self.capture=self.root/'capture'
        self.capture.mkdir()
        self.hypothesis=self.put('historical.json',{'fit_data':'original training only','training_count':3,
                'tag_side_m':.10,'edge_refined':{'T_base_camera':np.eye(4).tolist(),'T_hand_tag':np.eye(4).tolist()}})
        self.source=self.put('fit.json',{'per_sample':[{'image_stamp':i,'robot_tf_translation_m':[i,.21,0],
            'robot_tf_xyzw':[0,0,0,1]} for i in (1,2,3)],
            'matched_detector_frames':[{'image_stamp':i,'detector':'DICT_APRILTAG_36h11'} for i in (1,2,3)]})
        self.config=self.put('detector.json',{'object_corner_order_m':[[x*.05,y*.05,0] for x,y in
            ((-1,1),(1,1),(1,-1),(-1,-1))]})
        self.put('capture/manifest.json',{'arm':'right','effector':'right_gripper','frame':'base','tag_family':'tag36h11',
            'tag_id':0,'camera_frame':'color optical','camera_serial':'serial',
            'tag_axes':'canonical decoded corners 0->1 x, 0->3 y, x cross y z'})
        self.start=datetime(2026,11,20,tzinfo=timezone.utc).timestamp()+100

    def put(self,path,data):
        path=self.root/path
        path.write_text(json.dumps(data),encoding='utf-8')
        return path

    def row(self,i,x,error=0.):
        e=np.eye(4);e[0,3]=x
        c=e@np.diag([1.,-1.,-1.,1.]);c[1,3]+=error
        window=[]
        for j in range(7):
            stamp=self.start+2*i+.1*j
            window.append({'host_time_s':stamp,'endpoint_ros_stamp_s':stamp,'joint_ros_stamp_s':stamp,
                           'color_stamp_ms':stamp*1000,'depth_stamp_ms':stamp*1000,
                           'base_T_effector':e.tolist(),'camera_T_tag':c.tolist()})
        return dict(window[-1],arm='right',tag_family='tag36h11',tag_id=0,camera_serial='serial',
                    ruler_reference={'width_m':.1,'height_m':.1},width_m=.1,height_m=.1,stationary_frames=window,
                    frame_id='pose-'+str(i))

    def check(self, rows):
        self.put('capture/train.json',rows[:2])
        self.put('capture/holdout.json',rows[2:])
        return evaluate(self.hypothesis,self.source,self.config,[self.capture],'2026-11-20',10,3)

    def test_historical_pose_convention_both_axes_and_duplicate_close_points(self):
        result=self.check([self.row(1,.2),self.row(2,.24),self.row(3,.28)])
        self.assertEqual(result['distinct_new_poses'],3)
        self.assertTrue(result['consistency_passed'])
        self.assertAlmostEqual(result['samples'][0]['deg_if_axes_were_mixed'],180.)
        result=self.check([self.row(1,.2),self.row(2,.24),self.row(3,.243)])
        self.assertEqual(result['status'],'insufficient_distinct_poses')
        self.assertFalse(result['consistency_passed'])

    def test_near_duplicate_failures_remain_in_bounds_and_fit_stays_frozen(self):
        result=self.check([self.row(1,.2),self.row(2,.24),self.row(3,.28),self.row(4,.281,.03)])
        self.assertEqual(result['distinct_new_poses'],3)
        self.assertFalse(result['all_observations_within_limits'])
        self.assertEqual(result['current_samples_used_for_fit'],0)
        self.assertAlmostEqual(result['max_mm'],30.)
        self.assertIsNotNone(result['samples'][-1]['nearby_new_pose'])

    def test_repeat_stamp_in_wrong_split_or_unknown_convention_refuses(self):
        row=self.row(1,.2)
        with self.assertRaisesRegex(ValueError,'duplicated'):
            self.check([row,self.row(2,.25),row])
        self.put('detector.json',{'object_corner_order_m':[[0,0,0]]*4})
        with self.assertRaisesRegex(ValueError,'convention'):
            self.check([row])
