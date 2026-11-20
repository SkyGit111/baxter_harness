import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch

from cix_harness import ring_peg,ring_transport
from cix_harness.primitive_service import PrimitiveService
from tests.test_ring_peg import example


class HeldRingTransport(unittest.TestCase):
    def setUp(self):
        p,scene,state,ring,peg=example()
        self.profile,self.scene,self.state,self.ring,self.peg=p,scene,state,ring,peg
        p['ring_peg'].update(source_ring_id=ring['id'],destination_peg_id=peg['id'],transport={
            'via_tcp_base_m':[.680,.210,.09],'via_orientation_xyzw':[0,0,0,1],
            'peg_orientation_xyzw':[0,0,0,1],'above_peg_m':.073,'approach_m':.025,'payload_radius_m':.04})
        p['contact_band_m']=.025
        scene['held_ring_id']=ring['id'];ring['location']='held';state['gripping']=True
        robot=NS(cfg={'primitives':p,'max_observation_age_s':4,'planning':{'primitive_skills':['pick','place','move_to']}},
                 motion_remaining=30,joints=lambda:{'j':0.},audit=lambda *args,**kwargs:None)
        self.service=PrimitiveService(robot)

    def prepare(self,skill):
        self.service.observation=({'id':self.scene['obs_id'],'ring_scene':self.scene,'state':self.state},time.monotonic(),None)
        fake=lambda stages,*args,**kw:({'j':0.},[None]*len(stages))
        with patch.object(self.service,'_preflight',side_effect=fake):
            return self.service.prepare(self.scene['obs_id'],skill,uv=[.20,.45])

    def test_normal_prepare_requires_verified_matching_ring_and_routes_to_tip_without_gripper(self):
        receipt=self.prepare('move_to')
        self.assertEqual(receipt['status'],'ready',receipt)
        stages=receipt['stages'];self.assertFalse(any(s['kind']=='gripper' for s in stages))
        self.assertEqual(sum(s['kind']=='verify_grasp' for s in stages),2)
        self.assertTrue(receipt['ring_geometry']['held_transfer'])
        self.assertTrue(any(s.get('tcp_m')==[.680,.210,.09] for s in stages))
        self.assertEqual(stages[-2]['tcp_m'][:2],[.74,.27]);self.assertAlmostEqual(stages[-2]['tcp_m'][2],.163)
        self.assertEqual(self.prepare('place')['status'],'rejected')
        self.state['eef_m']=stages[-2]['tcp_m']
        self.assertTrue(ring_transport.at_hover(self.profile,self.state,self.peg))
        self.assertEqual(self.prepare('move_to')['status'],'rejected')
        self.ring['color']='blue'
        self.assertEqual(self.prepare('move_to')['status'],'rejected')

    def test_held_shape_obstacle_and_independent_post_move_verification(self):
        box={'id':'rod','frame':'base','xyz':[.74,.27,.1],'xyzw':[0,0,0,1],'size_m':[.02,.02,.02]}
        ring_transport.check_payload([.74,.27,.16],[box],.04)
        with self.assertRaisesRegex(ring_peg.RingGeometryError,'rod'):
            ring_transport.check_payload([.74,.27,.14],[box],.04)
        action={'skill':'move_to','obs_id':'earlier','ring_id':self.ring['id'],'peg_id':self.peg['id']}
        self.scene['held_transport']={'at_hover':True,'peg_id':self.peg['id']}
        self.assertEqual(ring_peg.verified(action,self.scene,self.state),'move_to')
        for change in ({'held_ring_id':None},{'obs_id':'earlier'}):
            self.assertIsNone(ring_peg.verified(action,dict(self.scene,**change),self.state))

    def test_near_hover_revised_peg_does_not_backtrack_to_the_pick_location(self):
        target=ring_transport.hover(self.profile,self.peg)
        self.state['eef_m']=[target[0]-.032,target[1]+.01,target[2]-.006]
        self.assertFalse(ring_transport.at_hover(self.profile,self.state,self.peg))
        self.assertTrue(ring_transport.near_peg(self.profile,self.state,self.peg))
        ring_transport.release_profile(self.profile,self.state,self.peg)
        receipt=self.prepare('move_to')
        self.assertEqual(receipt['status'],'ready',receipt)
        self.assertNotIn(self.profile['ring_peg']['transport']['via_tcp_base_m'],[s.get('tcp_m') for s in receipt['stages']])

    def test_fresh_fk_can_verify_transport_when_peg_is_occluded_but_cannot_open(self):
        target=ring_transport.hover(self.profile,self.peg)
        self.state['eef_m']=[target[0]-.006,target[1]-.004,target[2]-.002]
        self.service.ring_last_action={'skill':'move_to','peg_id':self.peg['id'],'ring_id':self.ring['id'],
                                      'obs_id':'earlier','contact_tcp_m':target}
        self.scene['pegs']=[]
        self.service.robot.camera=NS(last_depth=Mock(),intrinsics=Mock())
        obs={'id':'fresh','state':self.state}
        with patch.object(self.service,'_ring_scene',return_value=self.scene),patch('cix_harness.primitive_service.workspace_metadata'),patch.dict('sys.modules',{'pyrealsense2':Mock()}):
            self.service._describe_observation(obs)
        transport=obs['ring_scene']['held_transport']
        self.assertTrue(transport['at_hover']);self.assertFalse(transport['peg_currently_localized'])
        self.assertEqual(obs['ring_scene']['verified_skill'],'move_to')
        self.assertEqual(self.prepare('place')['status'],'rejected')

    def test_depth_uncertainty_applies_to_held_ring_but_not_fingertip_height(self):
        s=self.profile['ring_peg']
        s.update(matching_parts_verified=True,vertical_uncertainty_m=.0056,
                 release_gap_m=.004,fingertip_below_tcp_m=0.,
                 guided_release={'exposed_height_range_m':[.09,.10],'max_free_drop_m':.04})
        self.peg.update(tip_base_m=[.74,.27,-.05329],support_z_m=-.14791,support_uv_px=[40,70])
        self.ring['depth_plane_p90_m']=.00764
        h=ring_peg._guided_heights(s,self.ring,self.peg,-.04369,.013285)
        self.assertAlmostEqual(h['planned_tcp_z_m'],-.04369)
        self.assertAlmostEqual(h['free_drop_m'],.022885);self.assertAlmostEqual(h['free_drop_upper_m'],.036125)
        self.assertAlmostEqual(h['guided_distance_m'],.09462)
        self.assertEqual(ring_peg.resolve(self.profile,self.scene,self.state,'place',uv=[.40,.70])[1]['id'],self.peg['id'])
        with self.assertRaisesRegex(ring_peg.RingGeometryError,'free drop'):
            ring_peg._guided_heights(s,self.ring,self.peg,-.04369,.030)
        self.peg['support_z_m']=self.peg['tip_base_m'][2]-.0898
        uncertain=ring_peg._guided_heights(s,self.ring,self.peg,-.04369,.013285)
        self.assertLess(uncertain['guided_distance_m'],.09);self.assertGreater(uncertain['guided_distance_range_m'][1],.09)
        self.peg['support_z_m']=self.peg['tip_base_m'][2]-.06
        with self.assertRaisesRegex(ring_peg.RingGeometryError,'interval'):
            ring_peg._guided_heights(s,self.ring,self.peg,-.04369,.013285)

    def test_guided_release_servo_corrects_height_and_bounds_a_best_effort_trial(self):
        s=self.profile['ring_peg'];s.update(matching_parts_verified=True,vertical_uncertainty_m=.002,
            fingertip_below_tcp_m=0.,guided_release={'exposed_height_range_m':[.035,.045],
                'max_free_drop_m':.025,'matched_attempt_radius_fraction':.9})
        self.peg['radius_m']=.0095
        self.state['eef_m']=[.74,.27,.098]
        self.ring['center_base_m']=[.747,.27,.116];self.ring['depth_plane_p90_m']=.001
        prior={'peg_id':self.peg['id'],'ring_id':self.ring['id'],'color':'red'}
        def check(**args):
            return ring_peg.verify_before_open(self.profile,self.scene,self.state,prior,**args)
        with self.assertRaises(ring_peg.RingAlignmentNeeded):check()
        with self.assertRaises(ring_peg.RingAlignmentNeeded):check(servo_budget_spent=True,best_prior_offset_m=.003)
        result=check(servo_budget_spent=True,best_prior_offset_m=.008)
        self.assertEqual(result['release_basis'],'bounded_matching_parts_gravity_attempt')
        self.assertAlmostEqual(result['release_xy_error_m'],.007)
        self.ring['center_base_m'][0]=.751
        with self.assertRaises(ring_peg.RingAlignmentNeeded):check(servo_budget_spent=True,best_prior_offset_m=.012)
        self.ring['center_base_m']=[.74,.27,.109];self.state['eef_m'][2]=.091
        with self.assertRaises(ring_peg.RingAlignmentNeeded) as low:check()
        self.assertGreater(low.exception.dz,0.)
        self.ring['center_base_m'][2]=.133;self.state['eef_m'][2]=.115
        with self.assertRaises(ring_peg.RingAlignmentNeeded) as high:check()
        self.assertLess(high.exception.dz,0.)
        suffix=ring_peg.correction_suffix(self.profile,self.state,high.exception,[{'kind':'gripper','open':True}])
        self.assertEqual(suffix[0]['tcp_m'][:2],self.state['eef_m'][:2])
        self.assertLess(suffix[0]['tcp_m'][2],self.state['eef_m'][2])

    def test_registered_cap_with_no_depth_plane_uses_its_z_interval_once(self):
        import numpy as np
        s=self.profile['ring_peg'];s['vertical_uncertainty_m']=.0056
        self.assertAlmostEqual(ring_peg.measured({'depth_plane_p90_m':np.float32(.00766)},'depth_plane_p90_m'),.00766)
        with self.assertRaises(ring_peg.RingGeometryError):ring_peg.measured({'depth_plane_p90_m':True},'depth_plane_p90_m')
        peg=dict(self.peg,tip_base_m=[.74,.27,-.05329],depth_plane_p90_m=None,
                 tip_position_interval_base_m=[[.738,.271,-.05830],[.742,.269,-.04828]])
        self.assertEqual(ring_peg._tip_height_error(s,peg),(.0056,-.05830,-.04828))
        peg['depth_plane_p90_m']=.0065
        self.assertAlmostEqual(ring_peg._tip_height_error(s,peg)[0],.0065)
        peg['depth_plane_p90_m']=None;del peg['tip_position_interval_base_m']
        with self.assertRaises(ring_peg.RingGeometryError):ring_peg._tip_height_error(s,peg)


if __name__=='__main__':
    unittest.main()
