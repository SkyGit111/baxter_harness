import unittest

try:
    import cv2
    import numpy as np
except ImportError:
    cv2=None

from cix_harness.ring_peg import RingGeometryError


@unittest.skipIf(cv2 is None,'wrist feature test requires OpenCV')
class WristRegistration(unittest.TestCase):
    def test_oblique_height_projection_and_subpixel_circle_association(self):
        from cix_harness.ring_wrist import measure
        cfg={'K':[[350,0,110],[0,350,100],[0,0,1]],'D':[0.]*5,'resolution':[220,200],
             'eef_T_optical':{'xyz':[0.,0.,0.],'xyzw':[1.,0.,0.,0.]},
             'ring_hsv_ranges':{'blue':[[[110,160,90],[130,255,255]]]},
             'peg_tip_hsv_ranges':[[[0,0,210],[179,40,255]]],'relative_xy_uncertainty_m':.0003,
             'height_uncertainty_m':.0007,'cross_camera_tolerance_m':.005}
        state={'eef_m':[0.,0.,.6],'eef_xyzw':[0.,0.,0.,1.]}
        ring={'color':'blue','center_base_m':[.0037,0.,.512],'inner_radius_m':.007,'outer_radius_m':.014,'normal_base':[0,0,1]}
        peg={'tip_base_m':[0.,0.,.5],'radius_m':.0021,'axis_base':[0,0,1]};im=np.zeros((200,220,3),np.uint8)
        cv2.circle(im,(126,100),56,(255,0,0),-1);cv2.circle(im,(126,100),28,(0,0,0),-1)
        cv2.circle(im,(110,100),7,(255,255,255),-1)
        result=measure(im,state,ring,peg,cfg);self.assertAlmostEqual(result['offset_xy_m'][0],-.00402,places=4)
        self.assertLess(abs(result['offset_xy_m'][1]),.0001);self.assertLess(result['uncertainty_m'],.001)
        self.assertAlmostEqual(result['inner_radius_m'],.007,delta=.0003);self.assertAlmostEqual(result['peg_radius_m'],.0021,delta=.0003)
        cv2.circle(im,(110,100),8,(0,0,0),-1)
        with self.assertRaises(RingGeometryError):measure(im,state,ring,peg,cfg)
