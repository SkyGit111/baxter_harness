import unittest
from types import SimpleNamespace

import numpy as np

try:
    import cv2
except ImportError:
    cv2=None

from cix_harness.ring_perception import observe_ring_scene


@unittest.skipIf(cv2 is None,'Camera-side OpenCV runtime required')
class RingPerceptionTest(unittest.TestCase):
    def setUp(self):
        self.image=np.full((100,140,3),28,np.uint8)
        self.depth=np.ones((100,140),float)
        cv2.rectangle(self.image,(80,42),(125,62),(235,235,235),-1)
        y,x=np.indices(self.depth.shape)
        self.mask=(np.hypot(x-40,y-56)<=15)&(np.hypot(x-40,y-56)>6)
        self.image[self.mask]=(12,22,230);self.depth[self.mask]=.982
        self.intrinsics=SimpleNamespace(rays=lambda p:np.column_stack(((p[:,0]-70)/500,(p[:,1]-50)/500,np.ones(len(p)))))
        self.profile={'camera_to_base_xyzw':[1,0,0,0],'camera_to_base_translation_m':[0,0,1],
                      'table_plane':{'normal':[0,0,1],'offset':0},'tcp_offset_tool_m':[0,0,0]}
        self.settings={'board':{'roi_px':[75,35,130,68],'area_range_px':[800,1000]},
                       'rings':[{'id':'free_red','color':'red','roi_px':[15,30,65,81],'hsv_ranges':[[[0,80,80],[8,255,255]]]}]}

    def observation(self):
        return observe_ring_scene(self.image,self.depth,self.intrinsics,self.profile,self.settings,{'gripping':False})

    def test_measured_annular_top_and_loose_location_use_the_current_frame(self):
        obs=self.observation()
        self.assertIsNotNone(obs['board_polygon_px'])
        ring,=obs['rings']
        self.assertEqual(ring['location'],'loose')
        self.assertAlmostEqual(ring['outer_radius_m'],.029,delta=.002)
        self.assertAlmostEqual(ring['inner_radius_m'],.012,delta=.002)
        self.assertAlmostEqual(ring['center_base_m'][2],.018,places=5)
        self.assertGreater(ring['depth_inliers'],25)
        self.assertIsNone(obs['held_ring_id'])
        self.image[35:70,75:131]=28
        self.assertEqual(self.observation()['rings'][0]['location'],'unknown')

    def test_hole_and_depth_cannot_be_invented_from_colour_or_the_last_frame(self):
        self.assertTrue(self.observation()['rings'])
        self.depth[self.mask]=0
        self.assertFalse(self.observation()['rings'])
        self.depth[:]=.982
        cv2.circle(self.image,(40,56),15,(12,22,230),-1)
        self.assertFalse(self.observation()['rings'])
