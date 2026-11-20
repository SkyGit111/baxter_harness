import time
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

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


if __name__=='__main__':
    unittest.main()
