"""Mixed backend dispatch tests: fake robot only, no network or actuator access."""
import unittest
from types import SimpleNamespace as NS
from unittest.mock import Mock

from cix_harness.contract import ACTION_SPEC
from cix_harness.code_tool import CodeTool
from cix_harness.hybrid import backend_skills, run
from cix_harness.task_monitor import TaskMonitor


def health(scope='supervised_hybrid_task'):
    return {'motion_scope':scope,'hybrid_protocol':'serialized_backends_v1',
            'backend_capabilities':['primitives','vla'],'armed':True,
            'environment_fingerprint':'test-geometry','action_spec':ACTION_SPEC,
            'max_action_window':2,'max_action_segment':16,'motion_remaining':100}


class Robot:
    def __init__(self,scope='supervised_hybrid_task',lost=False):
        self.health=health(scope);self.calls=[];self.i=0;self.prepared=None;self.held=False;self.lost=lost
    def call(self,method,args=(),**kwargs):
        self.calls.append((method,args,kwargs))
        if method=='healthz':return dict(self.health)
        if method in ('primitive.observe','env.observe'):
            self.i+=1;self.prepared=None
            return {'id':str(self.i),'image':'test','state':{'eef_m':[.6,.2,.1],
                    'gripping':self.held,'gripper_percent':20 if self.held else 100},
                    'primitive_skills':['pick','place','clear_view','retreat'],
                    'recovery':{'view_indices':[0]}}
        if method=='primitive.prepare':
            assert args[0]==str(self.i)
            self.prepared=(str(self.i),args[1])
            return {'plan_id':str(self.i),'skill':args[1],'status':'ready','applied':False}
        if method=='primitive.execute':
            token,skill=self.prepared;assert token==args[0];self.prepared=None
            if self.lost:raise TimeoutError('reply lost after actuator dispatch')
            if skill=='pick':self.held=True
            if skill=='place':self.held=False
            return {'plan_id':token,'skill':skill,'status':'completed','applied':True,'gripping':self.held}
        if method=='env.step':
            ident,spec,action=args;assert ident==str(self.i);assert spec==ACTION_SPEC
            self.held=action[6]<.5
            return {'obs_id':ident,'applied':True,'gripping':self.held,
                    'gripper_percent':20 if self.held else 100,'endpoint_after_m':[.6,.2,.1]}
        raise AssertionError(method)


class Policy:
    def __init__(self,ready=True):self.calls=[];self.ready=ready
    def call(self,method,args=(),**kwargs):
        self.calls.append(method)
        if method=='healthz':return {'action_spec':ACTION_SPEC,'max_action_window':2,'motion_ready':self.ready}
        if method=='vla.predict':
            frame,options=args
            return {'obs_id':frame['id'],'action_spec':ACTION_SPEC,'actions':[[0,0,0,0,0,0,1]]*options['action_window']}
        raise AssertionError(method)


class Executive:
    def __init__(self,decisions):
        self.decisions=list(decisions);self.observations=[];self.feedback=None;self.monitor=TaskMonitor()
    def choose(self,obs):self.observations.append(obs);return self.decisions.pop(0)
    def recovery_allowed(self,reason):return True


def act(backend,skill,**kw):
    value={'decision':'act','backend':backend,'skill':skill,'subtask':skill+' the cyan block',
           'uv':[.5,.5] if backend=='primitives' and skill in ('pick','place') else None,
           'code':'','evidence':'current scene'}
    value.update(kw);return value


DONE={'decision':'done','reason':'none','skill':'','uv':None,'evidence':'verified'}


class Capabilities(unittest.TestCase):
    def test_old_profiles_never_gain_other_motion_backend(self):
        model={'action_spec':ACTION_SPEC,'motion_ready':True}
        self.assertEqual(set(backend_skills(health('supervised_primitive_task'),model,CodeTool())),{'primitives','code'})
        self.assertEqual(set(backend_skills(health('supervised_policy_task'),model,CodeTool())),{'vla','code'})
        unknown=health();unknown['hybrid_protocol']='not-a-real-protocol'
        self.assertEqual(backend_skills(unknown,model),{})

    def test_composite_requires_actual_policy_contract_and_readiness(self):
        model={'action_spec':ACTION_SPEC,'motion_ready':True}
        self.assertEqual(set(backend_skills(health(),model)),{'primitives','vla'})
        self.assertEqual(set(backend_skills(health(),{**model,'motion_ready':False})),{'primitives'})
        self.assertEqual(set(backend_skills(health(),{**model,'action_spec':'different'})),{'primitives'})


class Dispatch(unittest.TestCase):
    def args(self,execute=True):
        return NS(task='把青色块放到桌面',execute=execute,robot_port=9102,vla_port=9101,
                  max_turns=8,max_recoveries=3,max_grasp_retries=1,steps=1,
                  action_window=1,execution_mode='waypoints')

    def test_code_primitive_and_vla_share_serial_task_and_fresh_frames(self):
        robot=Robot();model=Policy();log=Mock()
        executive=Executive([act('code','compute',code='result = {"held": data["state"]["gripping"]}'),
                             act('primitives','pick'),act('vla','place'),DONE])
        self.assertEqual(run(self.args(),log,executive,robot,model,CodeTool()),0)
        self.assertEqual([c[0] for c in robot.calls if c[0] in ('primitive.execute','env.step')],
                         ['primitive.execute','env.step'])
        self.assertEqual([o['id'] for o in executive.observations],['1','2','3','5'])
        self.assertEqual(set(executive.observations[0]['backend_skills']),{'code','primitives','vla'})
        code=[c.kwargs for c in log.call_args_list if c.args[0]=='code_result'][0]
        self.assertFalse(code['applied']);self.assertEqual(code['result'],{'held':False})

    def test_selected_unavailable_backend_never_calls_motor_or_policy(self):
        robot=Robot('supervised_primitive_task');model=Policy()
        executive=Executive([act('vla','pick')])
        with self.assertRaisesRegex(RuntimeError,'not advertised'):
            run(self.args(),Mock(),executive,robot,model,CodeTool())
        self.assertEqual(model.calls,[])
        self.assertFalse(any(c[0] in ('primitive.execute','env.step') for c in robot.calls))

    def test_uncommissioned_vla_is_not_exposed_to_planner(self):
        robot=Robot();model=Policy(False);executive=Executive([DONE])
        self.assertEqual(run(self.args(),Mock(),executive,robot,model,CodeTool()),0)
        self.assertNotIn('vla',executive.observations[0]['backend_skills'])
        self.assertEqual(model.calls,['healthz'])

    def test_unknown_physical_outcome_never_falls_back_to_another_backend(self):
        robot=Robot(lost=True);model=Policy()
        executive=Executive([act('primitives','pick'),act('vla','pick')])
        with self.assertRaises(TimeoutError):run(self.args(),Mock(),executive,robot,model,CodeTool())
        self.assertEqual(sum(c[0]=='primitive.execute' for c in robot.calls),1)
        self.assertNotIn('vla.predict',model.calls)

    def test_primitive_and_vla_previews_never_execute_motors(self):
        for backend in ('primitives','vla'):
            robot=Robot();model=Policy();executive=Executive([act(backend,'pick')])
            self.assertEqual(run(self.args(False),Mock(),executive,robot,model,CodeTool()),0)
            self.assertFalse(any(c[0] in ('primitive.execute','env.step') for c in robot.calls))

    def test_compute_preview_has_no_physical_side_effect(self):
        robot=Robot();executive=Executive([act('code','compute',code='result = sum([1, 2, 3])')])
        self.assertEqual(run(self.args(False),Mock(),executive,robot,Policy(),CodeTool()),0)
        self.assertFalse(any(c[0] in ('primitive.prepare','primitive.execute','env.step') for c in robot.calls))

    def test_occluded_target_uses_available_primitive_recovery(self):
        robot=Robot();executive=Executive([{'decision':'blocked','reason':'occluded','evidence':'arm hides target'},
                                           act('primitives','pick'),DONE])
        self.assertEqual(run(self.args(),Mock(),executive,robot,Policy(),CodeTool()),0)
        self.assertEqual([c[1][1] for c in robot.calls if c[0]=='primitive.prepare'],['clear_view','pick'])


if __name__=='__main__':unittest.main()
