import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from cix_harness.ring_support_reference import check_reference, make_reference

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

    def test_a_dropped_ring_can_be_found_outside_its_original_roi(self):
        spec=self.settings['rings'][0]
        spec['loose_candidate']=True
        self.settings['table_polygon_px']=[[2,2],[137,2],[137,97],[2,97]]
        self.image[self.mask]=28;self.depth[self.mask]=1.
        y,x=np.indices(self.depth.shape)
        moved=(np.hypot(x-68,y-21)<=15)&(np.hypot(x-68,y-21)>6)
        self.image[moved]=(12,22,230);self.depth[moved]=.982
        ring,=self.observation()['rings']
        self.assertEqual(ring['location'],'loose')
        self.assertEqual(ring['localization'],'unique_free_ring_in_current_table_image')
        np.testing.assert_allclose(ring['center_uv_px'],[68,21],atol=1.)
        self.image[self.mask]=(12,22,230);self.depth[self.mask]=.982
        spec['roi_px']=[3,4,24,31]
        self.assertEqual(self.observation()['rings'],[])

    def test_a_single_round_depth_patch_cannot_replace_unseen_registered_post(self):
        self.settings['rings']=[]
        self.settings['pegs']=[{'id':'blue_peg','bracket_reference':{'tip_px':[40,56]}}]
        confused={'tip_uv_px':[48,56],'tip_base_m':[.03,.04,.09],'depth_inliers':19}
        registered={'id':'blue_peg','tip_uv_px':[40,56],
                    'tip_position_interval_base_m':[[.03,.04,.085],[.03,.04,.095]],
                    'current_tip_depth_measured':False}
        with patch('cix_harness.ring_perception._detect',return_value=[confused]), \
                patch('cix_harness.ring_perception._registered_peg',return_value=None):
            self.assertEqual(self.observation()['pegs'],[])
        with patch('cix_harness.ring_perception._detect',return_value=[confused]), \
                patch('cix_harness.ring_perception._registered_peg',return_value=registered):
            peg,=self.observation()['pegs']
            self.assertFalse(peg['current_tip_depth_measured'])
            confused['tip_uv_px']=[40,56]
            peg,=self.observation()['pegs']
            self.assertTrue(peg['current_tip_depth_measured'])
            self.assertEqual(peg['current_tip_depth_inliers'],19)

    def test_finger_approach_reference_requires_current_location_and_unchanged_clear_surface(self):
        color=np.full((16,18,3),30,np.uint8)
        depth=np.ones((16,18),float)
        ring={'center_uv_px':[9,8],'center_base_m':[.01,-.02,.01]}
        geom=SimpleNamespace(depth=depth,n=np.array([0.,0.,1.]),
                             rays=lambda px:np.tile([0.,0.,-1.],(len(px),1)))
        ref=make_reference([color]*5,[depth]*5,[0,0],np.ones(depth.shape,bool),ring,range(5))
        self.assertTrue(check_reference(geom,color,ring,ref)['verified'])
        color[2:6,2:6]=[60,80,10]
        self.assertFalse(check_reference(geom,color,ring,ref)['verified'])
        color[:]=30
        depth[2:6,2:6]=.992
        self.assertFalse(check_reference(geom,color,ring,ref)['verified'])
        depth[:]=1.
        moved=dict(ring,center_uv_px=[14,8])
        self.assertFalse(check_reference(geom,color,moved,ref)['verified'])
        depth[:]=.996
        self.assertFalse(check_reference(geom,color,ring,ref)['verified'])
