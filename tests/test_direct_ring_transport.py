import unittest

from cix_harness import ring_transport
from tests.test_ring_peg import example


class DirectRingTransport(unittest.TestCase):
    def test_optional_via_requires_both_fields_and_direct_move_keeps_the_ring_held(self):
        profile, scene, state, ring, peg=example()
        state['gripping']=True
        route={'peg_orientation_xyzw':[0.,0.,0.,1.],'above_peg_m':.06,
               'approach_m':.025,'payload_radius_m':.055}
        profile['ring_peg']['transport']=route
        ring_transport.validate(route)
        stages=ring_transport.stages(profile,state,peg)
        self.assertEqual([s for s in stages if s['kind']=='verify_grasp'],[{'kind':'verify_grasp','held':True}])
        self.assertFalse(any(s['kind']=='gripper' for s in stages))
        self.assertEqual(stages[-2]['tcp_m'],ring_transport.hover(profile,peg))
        with self.assertRaisesRegex(ValueError,'both position and orientation'):
            ring_transport.validate({**route,'via_tcp_base_m':[.59,.2,.14]})
