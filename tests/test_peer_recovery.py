"""A consumed, zero-command place plan can be replaced once after an observed peer drift."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from cix_harness.harness import run
from cix_harness.primitive_runner import PeerMotionRecovery, peer_moved_before_dispatch
from tests.test_executive import Journal, Robot, ScriptedPlanner, check, step


def refused():
    # Wire shape of the 2026-11-20 yellow cube place rejection on the dual bridge.
    return {'status':'failed','failure_code':'execution_failed','failure_type':'PeerStateChangedError',
            'message':'Inactive arm moved since observation; reobserve before motion','applied':False,
            'completed_stages':[],'completed_stage_count':0,'confirmed_commands':0,
            'failed_stage':{'stage_index':0,'phase':'transit','status':'failed','command_state':'not_dispatched'},
            'outcome_uncertain':False,'stop_attempted':False,'stopped':None,'stop_error':None,
            'recoverable':False,'retryable':False,'gripping':None}


def replay(folder,route,failures,policy='reobserve',max_recoveries=3):
    args=SimpleNamespace(task='将黄块放到桌上',route=route,execute=True,vlm_port=9110,vlm_model='replay',
        robot_port=9102,task_kind='place',max_turns=2,max_recoveries=max_recoveries,max_grasp_retries=0,
        max_replans=2,max_planner_calls=16,memory=str(Path(folder)/'memory.sqlite'))
    steps=[step('pick','yellow cube'),step('place','table',policy)]
    checks=[check(uv=[.7,.5]),check('step_done'),check(uv=[.60,.55]),check(uv=[.61,.56]),
            check('step_done'),check('task_done')]
    if route=='hybrid':
        for entry in steps:entry['backend']='primitives'
        for entry in checks:entry['code']=''
    planner=ScriptedPlanner([steps],checks)
    robot=Robot([None,*failures]);log=Journal(folder)
    return run(args,log,robot,planner),robot,log,planner


class PeerRecoveryTests(unittest.TestCase):
    def test_normal_entries_keep_verified_pick_and_replan_place_from_new_frame(self):
        for route in ('primitives','hybrid'):
            with self.subTest(route=route), tempfile.TemporaryDirectory() as folder:
                result,robot,log,planner=replay(folder,route,[refused()])
                self.assertEqual(result,0)
                self.assertEqual(robot.skills,['pick','place','place'])
                prepares=[args for method,args,_ in robot.calls if method=='primitive.prepare']
                self.assertEqual(prepares,[('1','pick',[.7,.5]),('2','place',[.60,.55]),('3','place',[.61,.56])])
                self.assertEqual([a[0] for m,a,_ in robot.calls if m=='primitive.execute'],['1','2','3'])
                self.assertEqual([e['step']['skill'] for k,e in log.events if k=='task_step_verified'],['pick','place'])
                self.assertEqual([e['plan_id'] for k,e in log.events if k=='peer_motion_replanned'],['3'])
                self.assertTrue(any(c.get('feedback',{}).get('failure_code')==PeerMotionRecovery.cause
                                for k,c in planner.contexts if k=='check' and c.get('feedback')))

    def test_second_drift_and_no_recovery_policies_stop_on_consumed_plan(self):
        for route in ('primitives','hybrid'):
            for failures,policy,budget,places in (([refused(),refused()],'reobserve',3,2),
                                                  ([refused()],'stop',3,1),([refused()],'reobserve',0,1)):
                with self.subTest(route=route,policy=policy,budget=budget),tempfile.TemporaryDirectory() as folder:
                    result,robot,log,_=replay(folder,route,failures,policy,budget)
                    self.assertEqual(result,2);self.assertEqual(robot.skills.count('place'),places)
                    self.assertEqual([e['step']['skill'] for k,e in log.events if k=='task_step_verified'],['pick'])

    def test_only_complete_unambiguous_no_dispatch_proof_qualifies(self):
        original=refused()
        self.assertTrue(peer_moved_before_dispatch(original))
        self.assertTrue(peer_moved_before_dispatch({**original,'failure_type':'PeerJointMovedError'}))
        for change in ({'applied':True},{'applied':None},{'confirmed_commands':1},{'confirmed_commands':False},
                       {'completed_stage_count':1},{'completed_stages':[{}]},{'outcome_uncertain':True},
                       {'stop_attempted':True},{'stop_error':'rpc lost'},{'arm_state_errors':{'left':'unavailable'}},
                       {'message':'Inactive arm occupancy changed or is unknown'},
                       {'failure_type':'MoveItCollisionError'},{'failure_type':'HeadCollisionError'},
                       {'failed_stage':{**original['failed_stage'],'stage_index':1}},
                       {'failed_stage':{**original['failed_stage'],'command_state':'outcome_unknown'}}):
            with self.subTest(change=change):self.assertFalse(peer_moved_before_dispatch({**original,**change}))
        for key in ('confirmed_commands','completed_stage_count','completed_stages','failed_stage','stop_attempted','outcome_uncertain'):
            partial=original.copy();del partial[key];self.assertFalse(peer_moved_before_dispatch(partial))

    def test_stale_observation_grasp_change_and_consumed_plan_cannot_dispatch(self):
        recovery=PeerMotionRecovery();log=Mock();before={'id':'a','state':{'arm':'left','gripping':True}}
        self.assertTrue(recovery.request({**refused(),'plan_id':'p'},before,1,None,log))
        self.assertIsNotNone(recovery.observation_error(before,log))
        self.assertIsNotNone(recovery.observation_error({'id':'b','state':{'arm':'left','gripping':False}},log))
        fresh={**before,'id':'b'};self.assertIsNone(recovery.observation_error(fresh,log))
        self.assertIsNotNone(recovery.plan_error({'plan_id':'p','obs_id':'b'},fresh,log))
        self.assertIsNotNone(recovery.plan_error({'plan_id':'q','obs_id':'a'},fresh,log))
        self.assertIsNone(recovery.plan_error({'plan_id':'q','obs_id':'b'},fresh,log))
