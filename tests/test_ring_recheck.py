import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from cix_harness.harness import run
from cix_harness.ring_recheck import _bound_ring,_fresh_match
from tests.test_executive import Journal,Robot,ScriptedPlanner,check,step


def record(n):
    ring={'id':'loose_blue','location':'loose','peg_id':None,'color':'blue','center_base_m':[.8457,.1346+.0006*n,-.2057],
          'outer_radius_m':.0152,'center_uv_px':[440,333],'rim_grasp_uv_px':[440,337],'local_table_upper_z_m':-.20246}
    return {'id':str(n),'state':{'arm':'left','gripping':False},'ring_scene':{'obs_id':str(n),'rings':[ring],'resolution':[640,480]}}


class RingRobot(Robot):
    def __init__(self,ready):super().__init__();self.ready=ready
    def call(self,method,args=(),**kwargs):
        result=super().call(method,args,**kwargs)
        if method=='primitive.observe':
            frame=record(self.i);result['ring_scene']=dict(frame['ring_scene'],verified_skill='pick' if self.held else None)
        if method=='primitive.prepare' and (self.i==1 or not self.ready):
            self.prepared=None
            result={'status':'rejected','obs_id':args[0],'skill':'pick','applied':False,'recoverable':True,'failure_code':'ring_geometry',
                    'message':'No fingertip contact depth above the local tabletop'}
        return result


class RingRecoveryTests(unittest.TestCase):
    def test_both_entries_recheck_mechanics_without_a_second_point_or_replan(self):
        for route in ('primitives','hybrid'):
            for ready in (True,False):
                with self.subTest(route=route,ready=ready),tempfile.TemporaryDirectory() as folder:
                    args=SimpleNamespace(task='抓起蓝环',route=route,execute=True,vlm_port=9110,vlm_model='fake',robot_port=9102,
                        task_kind='pick',max_turns=3,max_recoveries=2,max_grasp_retries=0,max_replans=1,max_planner_calls=9,memory=str(Path(folder)/'db'))
                    s=step('pick','blue ring');checks=[check(uv=[.6891,.7021])]
                    if ready:checks.extend((check('step_done'),check('task_done')))
                    if route=='hybrid':
                        s['backend']='primitives'
                        for c in checks:c['code']=''
                    planner=ScriptedPlanner([[s]],checks);robot=RingRobot(ready);journal=Journal(folder)
                    self.assertEqual(run(args,journal,robot,planner),0 if ready else 2)
                    selects=[a for m,a,_ in robot.calls if m=='primitive.prepare'];self.assertEqual(len(selects),2)
                    self.assertEqual(selects[1],('2','pick',[440/639,333/479]));self.assertEqual(robot.skills,['pick'] if ready else [])
                    self.assertEqual(sum(k=='plan' for k,c in planner.contexts),1)
                    self.assertEqual(sum(k=='check' for k,c in planner.contexts),3 if ready else 1)
                    if not ready:self.assertEqual([x['status'] for k,x in journal.events if k=='finish'],['ring_pick_geometry_unchanged'])

    def test_identity_drift_wrong_arm_unknown_or_applied_failures_do_not_authorize_recheck(self):
        original=record(1);new=record(2);choice={'skill':'pick','uv':[.6891,.7021]}
        bad={'obs_id':'1','status':'rejected','applied':False,'failure_code':'ring_geometry','recoverable':True,
             'geometry_independent_of_uv':True,'selected_ring_id':'loose_blue'}
        old=_bound_ring(bad,original,choice);self.assertIsNotNone(_fresh_match(old,original,new))
        for field,value in [('applied',None),('applied',True),('status','failed'),('failure_code','moveit_collision'),('selected_ring_id','red')]:
            self.assertIsNone(_bound_ring({**bad,field:value},original,choice))
        for field,value in [('id','red'),('location','held'),('outer_radius_m',.05),('center_base_m',[.90,.16,-.195])]:
            updated=copy.deepcopy(new);updated['ring_scene']['rings'][0][field]=value;self.assertIsNone(_fresh_match(old,original,updated))
        self.assertIsNone(_fresh_match(old,original,original));new['state']['arm']='right';self.assertIsNone(_fresh_match(old,original,new))
