import json,time,unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock
from cix_harness.primitive_service import PrimitiveService,IKPreflightError


class ContinuousPreflightTests(unittest.TestCase):
    def setUp(self):
        p=json.loads(Path('config/baxter-primitives.example.json').read_text())['primitives']
        p['calibrated']=True;p['tcp_workspace_min_m'][2]=-.17
        r=Mock();r.cfg={'primitives':p,'planning':{'trajectory_mode':'continuous'},
                       'joint_speed_rad_s':.03,'max_observation_age_s':3,
                       'commissioning_scope':'supervised_primitive_task'}
        r.armed=True;r.motion_remaining=8;r.joints.return_value={'j':0.}
        r.endpoint_position.return_value=[.6,.2,.1];r.arrival_error.return_value=(0.,0.)
        r.grip.gripping.return_value=False;r.move_segment.return_value={'elapsed_s':1.}
        r.continuous_ik.forward.side_effect=lambda joints:{'position':NS(x=.6,y=.2,z=.1),'orientation':NS(x=1,y=0,z=0,w=0)}
        self.r=r;self.s=PrimitiveService(r)
        self.path=[({'j':.1},NS())]

    def test_curve_interior_checked_even_with_valid_endpoints(self):
        def fk(joints):
            z=-.3 if .04<joints['j']<.06 else .1
            return {'position':NS(x=.6,y=.2,z=z),'orientation':NS(x=1,y=0,z=0,w=0)}
        self.r.continuous_ik.forward.side_effect=fk
        with self.assertRaises(IKPreflightError):self.s._check_curve({'j':0.},self.path,False,False,'transit')
        self.r.move_segment.assert_not_called()

    def test_executor_enables_curve_only_after_revalidation(self):
        stage={'kind':'move','tcp_m':[.6,.2,.1],'xyzw':[1,0,0,0]}
        self.s.prepared=('p',time.monotonic(),{'j':0.},[stage],[self.path],1,'move_to',False)
        receipt=self.s.execute('p')
        self.assertGreater(self.r.continuous_ik.forward.call_count,2)
        self.assertTrue(self.r.move_segment.call_args.kwargs['continuous'])
        self.assertEqual(receipt['completed_stages'][0]['motion_detail'],{'elapsed_s':1.})
