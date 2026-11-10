"""Offline fault-injection checks for recovery without any ROS/hardware connection."""
import copy
import json
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

from cix_harness.primitives import plan, validate_profile, TargetDepthError
from cix_harness.primitive_runner import run, validate_decision, decide, feedback
from cix_harness.primitive_service import PrimitiveService
from cix_harness.record_view import add_view


def profile():
    cfg=json.loads(Path('config/baxter-primitives.example.json').read_text())
    cfg['primitives']['observation_tcp_m']=[[.6,.4,.15],[.6,-.1,.15]]
    return cfg


def state(held=False):
    return {'arm':'left','eef_m':[.6,.2,.1],'eef_xyzw':[1,0,0,0],
            'gripping':held,'joint_names':['j'],'joint_rad':[0.]}


def decision(kind='act',skill='pick',reason='none'):
    return {'decision':kind,'reason':reason,'skill':skill,
            'uv':[.5,.5] if skill in ('pick','place','stack','move_to') else None,'evidence':'current image'}


class FakeRobot:
    def __init__(self,views=(0,1),outcomes=(),rejects=()):
        self.views=list(views);self.calls=[];self.serial=0;self.prepared=None
        self.outcomes=list(outcomes);self.rejects=list(rejects);self.held=False;self.skills=[]

    def call(self,method,args=(),**kwargs):
        self.calls.append((method,args,kwargs))
        if method=='healthz':return {'motion_scope':'supervised_primitive_task','armed':True}
        if method=='primitive.observe':
            self.serial+=1
            return {'id':str(self.serial),'image':'test','state':state(self.held),
                    'recovery':{'view_indices':self.views,'motion_remaining':64}}
        if method=='primitive.prepare':
            assert args[0]==str(self.serial), 'reused an old observation'
            self.prepared=(str(self.serial),args[1])
            outcome=self.rejects.pop(0) if self.rejects else None
            if outcome:
                return {'status':'rejected','applied':False,'recoverable':outcome!='stage_budget',
                        'failure_code':outcome,'message':outcome}
            return {'plan_id':str(self.serial),'status':'ready'}
        if method=='primitive.execute':
            assert args[0]==self.prepared[0], 'replayed a consumed plan'
            token,skill=self.prepared;self.prepared=None;self.skills.append(skill)
            outcome=self.outcomes.pop(0) if self.outcomes else None
            if isinstance(outcome,Exception):raise outcome
            if outcome=='empty_grasp':
                return {'plan_id':token,'applied':True,'status':'failed','failure_code':outcome,
                        'recoverable':True,'stopped':True,'gripping':False}
            if skill=='pick':self.held=True
            return {'plan_id':token,'applied':True,'status':'completed','gripping':self.held}
        raise AssertionError(method)


class RecoveryPlans(unittest.TestCase):
    def test_clear_view_lifts_then_moves_aside_and_keeps_held_object(self):
        p=profile()['primitives'];st=state(True)
        stages=plan('clear_view',st,p,view_index=1)
        self.assertEqual(stages[0]['tcp_m'],[.6,.2,.15])
        self.assertEqual(stages[1]['tcp_m'],[.6,-.1,.15])
        self.assertEqual(stages[-1],{'kind':'verify_grasp','held':True})
        self.assertFalse(any(s['kind']=='gripper' for s in stages))
        self.assertTrue(all(s['xyzw']==st['eef_xyzw'] for s in stages if s['kind']=='move'))

    def test_no_pose_no_guess_and_no_out_of_workspace_pose(self):
        p=profile()['primitives'];p['observation_tcp_m']=[]
        with self.assertRaisesRegex(ValueError,'No configured'):plan('clear_view',state(),p)
        for bad in ([2,0,.1],[.6,.2,-.1],[.6,float('nan'),.15]):
            p['observation_tcp_m']=[bad]
            with self.assertRaises(ValueError):validate_profile(p)

    def test_noop_view_and_dropping_object_are_rejected(self):
        p=profile()['primitives'];p['observation_tcp_m']=[[.6,.2,.1]]
        with self.assertRaisesRegex(ValueError,'Already at'):plan('clear_view',state(),p)
        with self.assertRaisesRegex(ValueError,'support'):plan('open',state(True),p)

    def test_recording_view_preserves_commissioning_flags(self):
        cfg=profile();original=copy.deepcopy(cfg)
        result=add_view(cfg,{'state':state()})
        self.assertEqual(cfg,original)
        self.assertFalse(result['commissioned']);self.assertFalse(result['primitives']['calibrated'])
        self.assertEqual(result['primitives']['observation_tcp_m'][-1],[.6,.2,.1])
        with self.assertRaisesRegex(ValueError,'already recorded'):add_view(result,{'state':state()})
        with self.assertRaisesRegex(ValueError,'arm'):add_view(cfg,{'state':{**state(),'arm':'right'}})


class RecoveryRunner(unittest.TestCase):
    def args(self,**kw):
        values={'task':'抓起方块','execute':True,'vlm_port':9110,'vlm_model':'local',
                'robot_port':9102,'max_turns':8,'max_recoveries':3,'max_grasp_retries':1,'task_kind':'pick'}
        values.update(kw)
        return NS(**values)

    def test_occlusion_clear_new_image_pick_verify(self):
        robot=FakeRobot();seen=[]
        answers=iter([decision('blocked','','occluded'),decision(),decision('done','')])
        def planner(port,model,task,obs,history):
            seen.append(obs['id']);return next(answers)
        self.assertEqual(run(self.args(),Mock(),robot,planner),0)
        self.assertEqual(robot.skills,['clear_view','pick']);self.assertEqual(seen,['1','2','3'])
        clear=next(c for c in robot.calls if c[0]=='primitive.prepare')
        self.assertEqual(clear[2]['kwargs'],{'view_index':0})

    def test_repeated_occlusion_tries_each_view_once_then_stops(self):
        robot=FakeRobot();planner=Mock(return_value=decision('blocked','','occluded'))
        self.assertEqual(run(self.args(),Mock(),robot,planner),2)
        self.assertEqual(robot.skills,['clear_view','clear_view'])
        self.assertEqual([c[2]['kwargs']['view_index'] for c in robot.calls if c[0]=='primitive.prepare'],[0,1])

    def test_missing_pose_or_ambiguous_target_never_moves(self):
        for views,reason in [((),'occluded'),((0,),'ambiguous_target'),((0,),'unsafe')]:
            robot=FakeRobot(views=views)
            self.assertEqual(run(self.args(),Mock(),robot,Mock(return_value=decision('blocked','',reason))),2)
            self.assertEqual(robot.skills,[])

    def test_preview_never_moves_even_for_occlusion(self):
        robot=FakeRobot()
        self.assertEqual(run(self.args(execute=False),Mock(),robot,Mock(return_value=decision('blocked','','occluded'))),0)
        self.assertEqual(robot.skills,[])

    def test_empty_grasp_retracts_before_replanning_and_never_replays(self):
        robot=FakeRobot(views=(),outcomes=['empty_grasp',None,None])
        planner=Mock(side_effect=[decision(),decision(),decision('done','')])
        self.assertEqual(run(self.args(),Mock(),robot,planner),0)
        self.assertEqual(robot.skills,['pick','retreat','pick'])
        self.assertEqual([c[1][0] for c in robot.calls if c[0]=='primitive.execute'],['1','2','3'])
        self.assertEqual([c.args[3]['id'] for c in planner.call_args_list],['1','3','4'])

    def test_second_empty_grasp_stops(self):
        robot=FakeRobot(views=(),outcomes=['empty_grasp',None,'empty_grasp'])
        self.assertEqual(run(self.args(),Mock(),robot,Mock(return_value=decision())),2)
        self.assertEqual(robot.skills,['pick','retreat','pick'])

    def test_empty_grasp_can_return_to_a_previously_successful_view(self):
        robot=FakeRobot(views=(0,),outcomes=[None,'empty_grasp',None,None])
        planner=Mock(side_effect=[decision('blocked','','occluded'),decision(),decision(),decision('done','')])
        self.assertEqual(run(self.args(),Mock(),robot,planner),0)
        self.assertEqual(robot.skills,['clear_view','pick','clear_view','pick'])

    def test_readonly_failures_reobserve_with_feedback(self):
        for error in ('target_depth','scene_changed','ik_preflight'):
            robot=FakeRobot(rejects=[error,None]);snapshots=[]
            answers=iter([decision(),decision(),decision('done','')])
            def planner(port,model,task,obs,history):
                snapshots.append(copy.deepcopy(history));return next(answers)
            self.assertEqual(run(self.args(),Mock(),robot,planner),0)
            self.assertEqual(robot.skills,['pick'])
            self.assertEqual(snapshots[1][-1]['failure_code'],error)

    def test_recovery_and_stage_budgets_bound_retries(self):
        for rejects,limit in [(['target_depth']*9,2),(['stage_budget'],0)]:
            robot=FakeRobot(rejects=rejects)
            self.assertEqual(run(self.args(max_recoveries=limit),Mock(),robot,Mock(return_value=decision())),2)
            self.assertEqual(robot.skills,[])
            self.assertEqual(sum(c[0]=='primitive.prepare' for c in robot.calls),limit+1)

    def test_lost_motion_reply_is_never_retried(self):
        robot=FakeRobot(outcomes=[TimeoutError('reply lost after dispatch')])
        with self.assertRaises(TimeoutError):run(self.args(),Mock(),robot,Mock(return_value=decision()))
        self.assertEqual(robot.skills,['pick'])

    def test_recovery_motion_alone_cannot_satisfy_pick(self):
        robot=FakeRobot()
        planner=Mock(side_effect=[decision('blocked','','occluded'),decision('done','')])
        self.assertEqual(run(self.args(),Mock(),robot,planner),2)
        self.assertEqual(robot.skills,['clear_view'])


class RecoveryProtocol(unittest.TestCase):
    def make_service(self):
        r=Mock();r.cfg=profile();r.cfg['max_observation_age_s']=3;r.motion_remaining=64
        r.joints.return_value={'j':0.}
        r.limb.endpoint_pose.return_value={'position':NS(x=.6,y=.2,z=.1),
                                          'orientation':NS(x=1.,y=0.,z=0.,w=0.)}
        r.continuous_ik.solve.return_value={'j':0.}
        service=PrimitiveService(r)
        service.observation=({'id':'o','state':state(True)},time.monotonic(),None)
        geometry=NS(PoseStamped=lambda:NS(header=NS(),pose=NS(position=NS(),orientation=NS())))
        modules={'geometry_msgs.msg':geometry,'tf':NS(transformations=NS(quaternion_slerp=lambda a,b,f:a))}
        return service,r,modules

    def test_clear_view_preflights_whole_path_without_opening_gripper(self):
        service,r,modules=self.make_service()
        with patch.dict('sys.modules',modules):result=service.prepare('o','clear_view',view_index=1)
        self.assertTrue(result['ik_preflight']);self.assertFalse(result['applied'])
        self.assertGreater(r.continuous_ik.solve.call_count,20)
        r.move_segment.assert_not_called();r.grip.command_position.assert_not_called()
        self.assertEqual(result['stages'][-1],{'kind':'verify_grasp','held':True})

    def test_model_mismatch_never_becomes_recoverable_ik_error(self):
        service,r,modules=self.make_service()
        r.continuous_ik.solve.side_effect=ValueError('KDL and measured endpoint disagree')
        with patch.dict('sys.modules',modules),self.assertRaisesRegex(ValueError,'disagree'):
            service.prepare('o','clear_view')
        self.assertIsNone(service.prepared);r.move_segment.assert_not_called()

    def test_stage_budget_checked_before_expensive_ik(self):
        service,r,modules=self.make_service();r.motion_remaining=2
        with patch.dict('sys.modules',modules):result=service.prepare('o','clear_view')
        self.assertEqual(result['failure_code'],'stage_budget');self.assertFalse(result['recoverable'])
        r.continuous_ik.solve.assert_not_called()

    def test_planner_receives_capabilities_and_structured_reason(self):
        d=decision('blocked','','occluded')
        obs={'state':state(),'image':'test','primitive_skills':['pick','clear_view'],'recovery':{'view_indices':[1]}}
        response={'choices':[{'finish_reason':'stop','message':{'content':json.dumps(d)}}]}
        with patch('cix_harness.primitive_runner.local_json',return_value=response) as http:
            self.assertEqual(decide(9110,'local','抓起方块',obs,[]),d)
            content=json.loads(http.call_args.args[2]['messages'][1]['content'][0]['text'])
        self.assertEqual(content['available_skills'],['pick','clear_view'])
        self.assertEqual(content['recovery']['view_indices'],[1])
        with self.assertRaises(ValueError):validate_decision({**d,'reason':'none'})

    def test_full_paths_are_not_sent_back_to_model(self):
        result=feedback(decision(),{'status':'completed','completed_stages':[{'eef_m':[1,2,3],'path':'x'*10000}]*9})
        self.assertNotIn('completed_stages',result);self.assertLess(len(json.dumps(result)),500)

    def test_depth_failure_consumes_observation_without_motion(self):
        r=Mock();r.cfg=profile();r.cfg['max_observation_age_s']=3
        service=PrimitiveService(r);service.observation=({'id':'o','state':state()},time.monotonic(),None)
        modules={'geometry_msgs.msg':NS(PoseStamped=Mock()),'tf':NS(transformations=Mock())}
        with patch.dict('sys.modules',modules),patch('cix_harness.primitive_service.ground_pixel',side_effect=TargetDepthError('hole')):
            result=service.prepare('o','pick',[.5,.5])
        self.assertEqual(result['failure_code'],'target_depth');self.assertIsNone(service.observation)
        self.assertIsNone(service.prepared);r.move_segment.assert_not_called()


if __name__=='__main__':unittest.main()
