import math
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

    def test_wrist_reobserves_held_annulus_when_main_view_is_occluded(self):
        from cix_harness.ring_wrist_held import held_ring
        from cix_harness.ring_peg import verified
        color=np.full((160,160,3),35,np.uint8)
        cv2.circle(color,(80,80),45,(20,45,235),-1)
        cv2.circle(color,(80,80),11,(40,40,40),-1)
        cfg={'resolution':[160,160],'K':[[400.,0.,80.],[0.,400.,80.],[0.,0.,1.]],'D':[0.]*5,
             'eef_T_optical':{'xyz':[0.,0.,-.14],'xyzw':[0.,0.,0.,1.]},
             'ring_hsv_ranges':{'red':[[[0,70,40],[10,255,255]]]},
             'surveyed_ring_sizes':{'r':{'inner_radius_m':.004,'outer_radius_m':.016,'outer_radius_uncertainty_m':.001}}}
        profile={'table_plane':{'normal':[0.,0.,1.],'offset':0.},'tcp_offset_tool_m':[0.,0.,0.],
                 'ring_peg':{'source_ring_id':'r','wrist_alignment':cfg,'perception':{
                     'rings':[{'id':'r','color':'red','loose_candidate':True}]}}}
        state={'gripping':True,'eef_m':[0.,0.,.10],'eef_xyzw':[1.,0.,0.,0.]}
        scene={'rings':[],'obs_id':'new'}
        row=held_ring(color,state,profile,scene)
        self.assertEqual(row['location'],'held')
        from cix_harness.ring_peg import fit_error
        self.assertGreater(fit_error(row,'depth_plane_p90_m'),0.)
        self.assertLess(math.dist(row['center_base_m'],state['eef_m']),.01)
        scene.update(held_ring_id='r',rings=[row])
        self.assertEqual(verified({'obs_id':'old','skill':'pick','ring_id':'r','color':'red'},scene,state),'pick')
        self.assertIsNone(held_ring(color,dict(state,gripping=False),profile,scene))
        self.assertIsNone(held_ring(color,state,profile,{'rings':[{'id':'r','location':'loose'}]}))
        from cix_harness.primitive_service import PrimitiveService
        service=PrimitiveService.__new__(PrimitiveService)
        camera=SimpleNamespace(last_color=color,last_depth=self.depth,last_number=20,
                               intrinsics=SimpleNamespace(width=160,height=160))
        service.robot=SimpleNamespace(camera=camera,wrist_condition=None)
        service.profile=profile
        service.ring_last_action=None
        missing=[{'id':'r','kind':'ring','detected_candidates':0},
                 {'id':'another','kind':'ring','detected_candidates':0}]
        with patch('cix_harness.ring_perception.observe_ring_scene',
                   return_value={'rings':[],'held_ring_id':None,'missing':missing}), \
                patch.object(service,'_fresh_ring_wrist',return_value=(color,{'frame_number':34})):
            observed=service._ring_scene(state,'after')
        self.assertEqual(observed['held_ring_id'],'r')
        self.assertEqual(observed['missing'],[missing[1]])
        self.assertEqual(observed['wrist_held_observation']['frame_number'],34)
        cv2.circle(color,(80,80),12,(20,45,235),-1)
        self.assertIsNone(held_ring(color,state,profile,scene))

    def test_coarse_board_window_becomes_a_peg_only_after_current_wrist_cap_and_hole_match(self):
        from cix_harness.primitive_service import PrimitiveService
        from cix_harness.ring_peg import RingGeometryError
        service=PrimitiveService.__new__(PrimitiveService)
        service.profile={'ring_peg':{'wrist_alignment':{'cross_camera_tolerance_m':.012}}}
        scene={'pegs':[],'rings':[{'id':'red','color':'red'}],'held_ring_id':'red','missing':[{'id':'post'}]}
        candidate={'id':'post','color':'red','tip_base_m':[.60,.02,.10],'radius_interval_m':[.003,.0046]}
        from importlib import import_module
        wrist=import_module('cix_harness.ring_wrist')
        match={'peg_radius_m':.0039,'peg_radius_error_m':.0006,'wrist_tip_base_m':[.601,.02,.10]}
        with patch.object(wrist,'measure',side_effect=RingGeometryError('cap occluded')):
            with self.assertRaises(RingGeometryError):service._confirm_wrist_peg(scene,{},candidate,None,{})
        self.assertEqual(scene['pegs'],[])
        with patch.object(wrist,'measure',return_value=dict(match,peg_radius_m=.011)):
            with self.assertRaises(RingGeometryError):service._confirm_wrist_peg(scene,{},candidate,None,{})
        self.assertEqual(scene['pegs'],[])
        with patch.object(wrist,'measure',return_value=match):
            service._confirm_wrist_peg(scene,{},candidate,None,{'frame_number':123})
        self.assertEqual(scene['pegs'][0]['measurement_source'],'registered_board_and_current_wrist_cap')
        self.assertEqual(scene['pegs'][0]['wrist_cap_frame']['frame_number'],123)
        self.assertEqual(scene['missing'],[])

    def test_fixed_cap_memory_requires_current_separated_rigid_mount_features(self):
        from cix_harness.ring_wrist_shaft import registered_cap
        rng=np.random.default_rng(9);image=rng.integers(0,255,(54,115,3),dtype=np.uint8)
        landmarks=[{'id':name,'uv_px':[x,25],'bgr':image[20:31,x-5:x+6].tolist()}
                   for name,x in [('blue',20),('corner',85)]]
        record={'frame':'base','method':'visible_cap_with_measured_protrusion','mount_anchors':landmarks,
                'observations':[{'arc_samples':16,'tip_interval_base_m':[[.6,.1,.08],[.6,.1,.09]]}]}
        self.assertEqual(len(registered_cap(record,image)),2)
        hidden=image.copy();hidden[18:34,75:95]=0
        self.assertIsNone(registered_cap(record,hidden))
        moved=np.roll(image,3,axis=1)
        self.assertIsNone(registered_cap(record,moved))
        self.assertIsNone(registered_cap(dict(record,observations=[]),image))
