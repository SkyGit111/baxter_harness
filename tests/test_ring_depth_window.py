import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from cix_harness.ring_support_reference import check_reference, make_reference, window_depth
from cix_harness.camera import D455Camera


class RingStationaryDepth(unittest.TestCase):
    def test_normal_camera_entry_keeps_last_rgb_depth_and_identifies_every_sample(self):
        camera=object.__new__(D455Camera)
        ids=iter([31,33,35,37,39])
        def sample():
            n=next(ids)
            camera.last_number=n
            camera.last_depth=np.full((3,3),.001*n+.70)
            camera.last_pair={'depth_frame_number':n+10}
            return n
        camera._aligned_pair=Mock(side_effect=sample)
        def last():
            return 'latest image',2.5,sample()
        camera.capture=Mock(side_effect=last)
        image, timestamp, frame=camera.capture_ring_depth()
        self.assertEqual((image,timestamp,frame),('latest image',2.5,39))
        self.assertEqual(camera.capture.call_count,1)
        self.assertEqual(camera._aligned_pair.call_count,4)
        window=camera.ring_depth_window
        self.assertEqual(window['frame_numbers'],[31,33,35,37,39])
        self.assertEqual(window['depth_frame_numbers'],[41,43,45,47,49])
        self.assertAlmostEqual(float(window['depth_m'][0,0]),.735)
        self.assertAlmostEqual(float(camera.last_depth[0,0]),.739)

    def test_requires_four_observed_depths_and_uses_the_correct_median(self):
        frames = [np.array([a,b,c]) for a,b,c in
                  [(0.,0.,1.10),(0.,1.21,1.11),(1.5,1.22,1.12),(1.31,1.23,1.14),(1.32,1.24,1.14)]]
        np.testing.assert_allclose(window_depth(frames), [0.,1.225,1.12])

    def test_newest_obstacle_supersedes_clear_median(self):
        try:
            import cv2
        except ImportError:
            self.skipTest('OpenCV is required for camera footprint comparison')
        depth = np.ones((18,18))
        image = np.full((18,18,3),50,dtype=np.uint8)
        geom = SimpleNamespace(depth=depth, n=np.array([0.,0.,1.]),
                               rays=lambda xy: np.tile([0.,0.,-1.],(len(xy),1)))
        ring = {'center_uv_px':[8.,8.],'center_base_m':[.6,.2,0.]}
        ref = make_reference([image]*5,[depth]*5,[0,0],np.ones(depth.shape,bool),ring,[1,2,3,4,5])
        arriving = depth.copy()
        arriving[2:6,2:6] -= .009
        self.assertTrue(check_reference(geom,image,ring,ref,[depth,depth])['verified'])
        failure=check_reference(geom,image,ring,ref,[arriving,arriving])
        self.assertEqual(failure['reason'], 'new_obstacle_or_occlusion_in_latest_depth_pairs')
        arriving[2:6,2:6] -= .009
        self.assertFalse(check_reference(geom,image,ring,ref,[depth,arriving])['verified'])
