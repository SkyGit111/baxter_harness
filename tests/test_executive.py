"""Task-level integration and memory provenance, using only simulated executors."""
import copy
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from cix_harness.executive import TaskExecutive
from cix_harness.harness import run
from cix_harness.memory import EpisodeMemory, geometry_fingerprint, scope_key
from cix_harness.task_plan import EXPECTED, validate_plan, validate_check, VisualTaskPlanner
from cix_harness.contract import ACTION_SPEC


def step(skill='pick', target='cyan cube', policy='reobserve'):
    return {'skill': skill, 'target': target, 'instruction': skill+' the '+target,
            'expect': EXPECTED[skill], 'on_failure': policy}


def check(kind='act', uv=None, reason='none'):
    return {'decision': kind, 'reason': reason, 'uv': uv, 'evidence': 'Current scene matches target'}


def obs(i='1', held=False, z=.1):
    return {'id': str(i), 'image': 'synthetic', 'state': {'arm': 'left', 'eef_m': [.6,.2,z],
            'gripping': held, 'gripper_percent': 20 if held else 100},
            'primitive_skills': ['pick','place','stack','retreat','clear_view'],
            'recovery': {'view_indices': [0]}}


class Journal:
    def __init__(self, folder): self.folder=Path(folder); self.events=[]
    def __call__(self, name, **fields): self.events.append((name,copy.deepcopy(fields)))
    def camera(self, observation, role): self('observation', id=observation['id'], role=role)


class ScriptedPlanner:
    def __init__(self, plans, checks):
        self.plans=list(plans);self.checks=list(checks);self.contexts=[]
    def plan(self, obs, context):
        self.contexts.append(('plan',copy.deepcopy(context)))
        return {'steps':self.plans.pop(0)}
    def check(self, obs, context):
        self.contexts.append(('check',copy.deepcopy(context)))
        return self.checks.pop(0)


class Robot:
    def __init__(self, outcomes=()):
        self.calls=[];self.skills=[];self.i=0;self.held=False;self.prepared=None;self.outcomes=list(outcomes)
    def call(self, method, args=(), **kwargs):
        self.calls.append((method,args,kwargs))
        if method=='healthz':
            return {'armed':True,'arm':'left','motion_scope':'supervised_primitive_task','environment_fingerprint':'test-profile'}
        if method=='primitive.observe':
            self.i+=1;return obs(self.i,self.held)
        if method=='primitive.prepare':
            assert args[0]==str(self.i)
            self.prepared=(args[0],args[1]);return {'plan_id':args[0],'skill':args[1],'status':'ready'}
        if method=='primitive.execute':
            assert self.prepared[0]==args[0]
            token,skill=self.prepared;self.prepared=None;self.skills.append(skill)
            outcome=self.outcomes.pop(0) if self.outcomes else None
            if isinstance(outcome,Exception):raise outcome
            if outcome=='empty_grasp':
                return {'plan_id':token,'skill':skill,'applied':True,'status':'failed','failure_code':outcome,
                        'recoverable':True,'stopped':True,'gripping':False}
            if skill=='pick':self.held=True
            if skill in ('place','stack'):self.held=False
            return {'plan_id':token,'skill':skill,'applied':True,'status':'completed','gripping':self.held}
        raise AssertionError(method)


class ManagedTasks(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        self.log=Journal(self.temp.name)
        self.args=NS(task='把青色方块叠到黄色方块上',route='primitives',execute=True,vlm_port=9110,
                     vlm_model='test',robot_port=9102,task_kind='stack',max_turns=8,max_recoveries=3,
                     max_grasp_retries=1,max_replans=2,max_planner_calls=32,
                     memory=str(Path(self.temp.name)/'memory.sqlite'))

    def test_full_stack_keeps_plan_checkpoints_and_retrievable_memory(self):
        planner=ScriptedPlanner([[step(),step('stack','yellow cube')]],
            [check(uv=[.4,.5]),check('step_done'),check(uv=[.7,.5]),check('step_done'),check('task_done')])
        robot=Robot()
        self.assertEqual(run(self.args,self.log,robot,planner),0)
        self.assertEqual(robot.skills,['pick','stack'])
        snapshot=json.loads((self.log.folder/'task-state.json').read_text(encoding='utf-8'))
        self.assertTrue(snapshot['task_verified']);self.assertEqual(len(snapshot['verified']),2)
        self.assertEqual([s['status'] for s in snapshot['steps']],['verified','verified'])
        memory=EpisodeMemory(self.args.memory);self.addCleanup(memory.close)
        found=memory.retrieve(self.args.task,scope_key(robot.call('healthz'),'primitives','test'))
        self.assertEqual(found[0]['outcome'],'verified');self.assertEqual(len(found[0]['recipe']),2)
        self.assertNotIn('uv',json.dumps(found));self.assertNotIn('eef_m',json.dumps(found))

    def test_preview_cannot_be_learned_as_success(self):
        self.args.execute=False
        planner=ScriptedPlanner([[step(),step('stack','yellow cube')]],[check(uv=[.4,.5])])
        robot=Robot();self.assertEqual(run(self.args,self.log,robot,planner),0)
        self.assertEqual(robot.skills,[])
        memory=EpisodeMemory(self.args.memory);self.addCleanup(memory.close)
        self.assertEqual(memory.retrieve(self.args.task,scope_key(robot.call('healthz'),'primitives','test')),[])
        self.assertEqual(memory.db.execute('SELECT mode,status FROM episodes').fetchone(),('preview','preview'))

    def test_empty_grasp_retraction_then_replan_retains_original_goal(self):
        self.args.task_kind='pick';self.args.task='抓起青色方块'
        planner=ScriptedPlanner([[step(policy='replan')],[step(target='cyan cube on image right')]],
            [check(uv=[.4,.5]),check(uv=[.6,.5]),check('step_done'),check('task_done')])
        robot=Robot(['empty_grasp',None,None])
        self.assertEqual(run(self.args,self.log,robot,planner),0)
        self.assertEqual(robot.skills,['pick','clear_view','pick'])
        plans=[c for kind,c in planner.contexts if kind=='plan']
        self.assertEqual(plans[0]['goal'],plans[1]['goal']);self.assertEqual(plans[1]['cause'],'empty_grasp')
        snapshot=json.loads((self.log.folder/'task-state.json').read_text(encoding='utf-8'))
        self.assertEqual(len(snapshot['verified']),1);self.assertEqual(snapshot['replans'],1)
        self.assertIn('empty_grasp',snapshot['failures'])

    def test_stop_policy_suppresses_automatic_grasp_recovery(self):
        planner=ScriptedPlanner([[step(policy='stop')]],[check(uv=[.4,.5])]);robot=Robot(['empty_grasp'])
        self.assertEqual(run(self.args,self.log,robot,planner),2)
        self.assertEqual(robot.skills,['pick'])

    def test_occlusion_recovery_does_not_complete_the_pick_step(self):
        self.args.task_kind='pick';self.args.task='抓起青色方块'
        planner=ScriptedPlanner([[step()]], [check('blocked',reason='occluded'),check(uv=[.5,.5]),
                                          check('step_done'),check('task_done')])
        robot=Robot();self.assertEqual(run(self.args,self.log,robot,planner),0)
        self.assertEqual(robot.skills,['clear_view','pick'])
        verifies=[fields for name,fields in self.log.events if name=='task_step_verified']
        self.assertEqual(len(verifies),1);self.assertEqual(verifies[0]['step']['skill'],'pick')

    def test_final_goal_verification_can_also_clear_occlusion(self):
        self.args.task_kind='pick';self.args.task='抓起青色方块'
        planner=ScriptedPlanner([[step()]], [check(uv=[.5,.5]),check('step_done'),
                                          check('blocked',reason='occluded'),check('task_done')])
        robot=Robot();self.assertEqual(run(self.args,self.log,robot,planner),0)
        self.assertEqual(robot.skills,['pick','clear_view'])

    def test_unknown_execution_is_not_retried_or_retrievable(self):
        planner=ScriptedPlanner([[step()]],[check(uv=[.4,.5])]);robot=Robot([TimeoutError('lost after dispatch')])
        with self.assertRaises(TimeoutError):run(self.args,self.log,robot,planner)
        self.assertEqual(robot.skills,['pick'])
        memory=EpisodeMemory(self.args.memory);self.addCleanup(memory.close)
        self.assertEqual(memory.db.execute('SELECT status FROM episodes').fetchone()[0],'unknown')
        self.assertEqual(memory.retrieve(self.args.task,scope_key(robot.call('healthz'),'primitives','test')),[])

    def test_replanning_preserves_verified_prefix(self):
        planner=ScriptedPlanner([[step(),step('stack','yellow cube')],[step('stack','yellow cube on image left')]],
            [check(uv=[.4,.5]),check('step_done'),check('replan',reason='ambiguous_target'),
             check(uv=[.2,.5]),check('step_done'),check('task_done')])
        robot=Robot();self.assertEqual(run(self.args,self.log,robot,planner),0)
        self.assertEqual(robot.skills,['pick','stack'])
        plans=[c for kind,c in planner.contexts if kind=='plan']
        self.assertEqual(plans[1]['verified'],[{'skill':'pick','target':'cyan cube'}])

    def test_shared_executive_runs_through_existing_vla_rollout(self):
        self.args.route='vla';self.args.task_kind='pick';self.args.task='抓起青色方块'
        self.args.vla_port=9101;self.args.steps=1;self.args.action_window=1
        self.args.execution_mode='waypoints';self.args.planner_view='scene'
        robot=Robot();position=[.6,.2,.1];count=[0];commands=[]
        def robot_call(method,args=(),**kwargs):
            if method=='healthz':return {'armed':True,'arm':'left','motion_scope':'supervised_policy_task',
                                        'action_spec':ACTION_SPEC,'environment_fingerprint':'test-vla'}
            if method=='env.observe':
                count[0]+=1;return obs(count[0],robot.held,position[2])
            if method=='env.step':
                ident,spec,action=args;self.assertEqual(ident,str(count[0]));commands.append(ident)
                if robot.held:position[2]+=.06
                robot.held=True
                return {'obs_id':ident,'applied':True,'gripping':True,'gripper_percent':20,'endpoint_after_m':list(position)}
            raise AssertionError(method)
        robot.call=robot_call
        def policy_call(method,args=(),**kw):
            self.assertEqual(method,'vla.predict');frame,options=args
            return {'obs_id':frame['id'],'action_spec':ACTION_SPEC,'actions':[[0,0,.06,0,0,0,0]]}
        policy=NS(call=policy_call)
        planner=ScriptedPlanner([[step()]],[check(),check(),check('step_done'),check('task_done')])
        with patch('cix_harness.__main__.check_models',return_value=(policy,{'action_spec':ACTION_SPEC})), \
             patch('cix_harness.__main__.HttpRpcClient',return_value=robot):
            self.assertEqual(run(self.args,self.log,robot,planner),0)
        self.assertEqual(len(commands),2)
        snapshot=json.loads((self.log.folder/'task-state.json').read_text(encoding='utf-8'))
        self.assertTrue(snapshot['task_verified']);self.assertEqual(snapshot['verified'][0]['skill'],'pick')

    def test_live_memory_is_present_in_next_task_plan_context(self):
        self.args.task_kind='pick';self.args.task='抓起青色方块'
        robot=Robot();memory=EpisodeMemory(self.args.memory)
        episode=memory.start(scope_key(robot.call('healthz'),'primitives','test'),'primitives',self.args.task,'live')
        memory.finish(episode,'verified',[step()],['occluded'],['clear_view']);memory.close()
        self.args.execute=False
        planner=ScriptedPlanner([[step()]],[check(uv=[.4,.5])])
        self.assertEqual(run(self.args,self.log,robot,planner),0)
        context=planner.contexts[0][1]
        self.assertEqual(context['memory'][0]['recovery_attempts'],['clear_view'])
        self.assertEqual(context['memory'][0]['recipe'][0]['skill'],'pick')


class ContractsAndEvidence(unittest.TestCase):
    def executive(self, route, planner, **kw):
        memory=EpisodeMemory(':memory:');self.addCleanup(memory.close)
        return TaskExecutive('抓起青色方块',route,planner,memory,None,Mock(),mode='live',**kw)

    def test_invalid_skill_sequence_and_wrong_expect_are_rejected(self):
        for steps in ([step('stack')],[step(),step()],[{**step(),'expect':'stacked'}],[step('clear_view')]):
            with self.assertRaises(ValueError):validate_plan({'steps':steps},['pick','stack'],False)
        with self.assertRaises(ValueError):validate_check(check(uv=[1.1,.2]),'primitives',step())

    def test_done_and_step_done_before_execution_cannot_pass(self):
        for kind in ('task_done','step_done'):
            planner=ScriptedPlanner([[step()]],[check(kind),check(kind)])
            e=self.executive('primitives',planner)
            self.assertEqual(e.choose(obs())['decision'],'blocked')
            self.assertFalse(e.task_verified);self.assertEqual(e.verified,[])

    def test_vla_requires_measured_grasp_and_lift_not_acknowledgement(self):
        planner=ScriptedPlanner([[step()]],[check(),check('step_done'),check('step_done')])
        e=self.executive('vla',planner);self.assertEqual(e.choose(obs())['subtask'],'pick the cyan cube')
        e.event('prediction',{'obs_id':'2','actions':[[0,0,0,0,0,0,0]]})
        e.event('applied',{'obs_id':'2','applied':True,'endpoint_after_m':[.6,.2,.1],'gripping':True,'gripper_percent':20})
        self.assertEqual(e.choose(obs('3',True,.1))['decision'],'blocked')
        self.assertEqual(e.verified,[])

    def test_vla_verified_lift_allows_step_then_task_completion(self):
        planner=ScriptedPlanner([[step()]],[check(),check('step_done'),check('task_done')])
        e=self.executive('vla',planner);e.choose(obs())
        for ident,z in [('2',.1),('3',.16)]:
            e.event('prediction',{'obs_id':ident,'actions':[[0,0,.06,0,0,0,0]]})
            e.event('applied',{'obs_id':ident,'applied':True,'endpoint_after_m':[.6,.2,z],'gripping':True,'gripper_percent':20})
        self.assertEqual(e.choose(obs('4',True,.16))['decision'],'done')
        self.assertTrue(e.task_verified)

    def test_downward_motion_does_not_verify_a_vla_lift(self):
        planner=ScriptedPlanner([[step('lift')]], [check(),check('step_done'),check('step_done')])
        e=self.executive('vla',planner);e.choose(obs())
        e.event('prediction',{'obs_id':'2','actions':[[0,0,-.08,0,0,0,1]]})
        e.event('applied',{'obs_id':'2','applied':True,'endpoint_after_m':[.6,.2,.02],
                           'gripping':False,'gripper_percent':100})
        self.assertEqual(e.choose(obs('3',False,.02))['decision'],'blocked')
        self.assertEqual(e.verified,[])

    def test_final_completion_rechecks_grasp_after_last_step_verified(self):
        planner=ScriptedPlanner([[step()]], [check(uv=[.5,.5]),check('step_done'),
                  check('blocked',reason='occluded'),check('task_done'),check('task_done')])
        e=self.executive('primitives',planner);e.choose(obs())
        e.event('primitive_result',{'skill':'pick','status':'completed','applied':True})
        self.assertEqual(e.choose(obs('2',True))['decision'],'blocked')
        self.assertEqual(len(e.verified),1)
        self.assertEqual(e.choose(obs('3',False))['decision'],'blocked')
        self.assertFalse(e.task_verified)

    def test_replan_budget_and_planner_budget_are_enforced(self):
        planner=ScriptedPlanner([[step()]],[check('replan',reason='no_progress')])
        e=self.executive('primitives',planner,max_replans=0)
        self.assertEqual(e.choose(obs())['decision'],'blocked')
        planner=ScriptedPlanner([[step()]],[check()])
        e=self.executive('vla',planner,max_calls=1)
        with self.assertRaisesRegex(RuntimeError,'call budget'):e.choose(obs())

    def test_model_request_uses_json_plan_schema_and_current_image(self):
        planner=VisualTaskPlanner(9110,'local')
        response={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({'steps':[step()]})}}]}
        with patch('cix_harness.task_plan.local_json',return_value=response) as http:
            planner.plan(obs(),{'goal':'抓青色方块'})
        body=http.call_args.args[2]
        self.assertEqual(body['response_format']['json_schema']['name'],'task_plan')
        self.assertEqual(body['messages'][1]['content'][1]['image_url']['url'],'synthetic')


class MemoryIsolation(unittest.TestCase):
    def test_scope_provenance_age_and_status_filtering(self):
        memory=EpisodeMemory(':memory:');self.addCleanup(memory.close)
        for mode,status,scope in [('live','verified','a'),('preview','verified','a'),('simulation','verified','a'),
                                  ('live','unknown','a'),('live','verified','b')]:
            episode=memory.start(scope,'primitives','抓起青色方块',mode)
            memory.finish(episode,status,[step()],[])
        found=memory.retrieve('抓起青色方块','a')
        self.assertEqual(len(found),1)
        self.assertEqual(memory.retrieve('抓起青色方块',None),[])
        memory.db.execute('UPDATE episodes SET created=?',(time.time()-86400*31,));memory.db.commit()
        self.assertEqual(memory.retrieve('抓起青色方块','a'),[])

    def test_geometry_change_changes_scope_but_runtime_budget_does_not(self):
        cfg={'arm':'left','primitives':{'table_z_m':-.18},'motion_budget':10}
        one=geometry_fingerprint(cfg)
        self.assertEqual(one,geometry_fingerprint({**cfg,'motion_budget':30}))
        self.assertNotEqual(one,geometry_fingerprint({**cfg,'primitives':{'table_z_m':-.16}}))
        self.assertIsNone(scope_key({'arm':'left'},'primitives','model'))


if __name__=='__main__':unittest.main()
