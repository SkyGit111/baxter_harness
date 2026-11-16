import copy
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from cix_harness.harness import run
from cix_harness.task_plan import validate_plan
from test_executive import Journal, ScriptedPlanner, step, check, obs


def arm_step(arm,skill):return dict(step(skill),arm=arm,backend='primitives',instruction=skill.replace('_',' ')+' the cyan cube')
def answer(kind='act',uv=None):return dict(check(kind,uv),code='')


class DualRobot:
    def __init__(self,lost=False):self.i=0;self.held={'left':False,'right':False};self.token=None;self.calls=[];self.lost=lost
    def call(self,method,args=(),kwargs=None,**options):
        self.calls.append((method,args))
        if method=='healthz':return {'armed':True,'dual_protocol':'serialized_dual_primitives_v1',
            'available_arms':['left','right'],'arm':'both','environment_fingerprint':'dual','motion_scope':'supervised_dual_primitive_task'}
        arm=args[0]
        if method=='dual.observe':
            self.i+=1;self.token=None
            result=obs(self.i,self.held[arm]);result['state']['arm']=arm
            result.update(available_arms=['left','right'],selected_arm=arm,
                arm_states={a:dict(obs(self.i,self.held[a])['state'],arm=a) for a in self.held})
            return result
        if method=='dual.prepare':
            self.token=(arm,str(self.i),args[2]);assert args[1]==str(self.i)
            return {'status':'ready','plan_id':str(self.i),'skill':args[2],'applied':False,'selected_arm':arm}
        if method=='dual.execute':
            a,t,s=self.token;self.token=None;assert (a,t)==(arm,args[1])
            if self.lost:raise TimeoutError('unknown motor receipt')
            if s=='pick':self.held[arm]=True
            if s=='place':self.held[arm]=False
            return {'status':'completed','applied':True,'plan_id':t,'skill':s,'gripping':self.held[arm],'selected_arm':arm}
        raise AssertionError(method)


class ArmTasks(unittest.TestCase):
    def setup_task(self,folder,arms='both'):
        return NS(task='左手放到桌面后右手抓起',route='hybrid',arms=arms,execute=True,
                  planner_provider='local',vlm_model='fake',vlm_port=9110,robot_port=9102,vla_port=9101,
                  max_turns=8,max_recoveries=2,max_grasp_retries=1,max_replans=2,max_planner_calls=24,
                  memory=str(Path(folder)/'memory.sqlite3'))

    def test_two_arm_table_transfer_uses_new_observation_on_switch(self):
        with tempfile.TemporaryDirectory() as folder:
            robot=DualRobot();robot.held['left']=True
            planner=ScriptedPlanner([[arm_step('left','place'),arm_step('right','pick')]],
                [answer(uv=[.5,.5]),answer('step_done'),answer(uv=[.5,.5]),answer('step_done'),answer('task_done')])
            log=Journal(folder)
            self.assertEqual(run(self.setup_task(folder),log,robot,planner),0)
            self.assertEqual(robot.held,{'left':False,'right':True})
            executed=[a for m,a in robot.calls if m=='dual.execute']
            self.assertEqual([a[0] for a in executed],['left','right'])
            self.assertNotEqual(executed[0][1],executed[1][1])
            import json
            snapshot=json.loads((Path(folder)/'task-state.json').read_text())
            self.assertTrue(snapshot['task_verified'])
            self.assertEqual([s['arm'] for s in snapshot['verified']],['left','right'])
            timings=[f for name,f in log.events if name=='component_timing' and f['component']=='robot_rpc']
            self.assertEqual(len(timings),len(robot.calls))

    def test_holding_preconditions_belong_to_each_arm(self):
        states={'left':{'gripping':True},'right':{'gripping':False}}
        validate_plan({'steps':[arm_step('right','pick'),arm_step('left','place')]},['pick','place'],True,{'primitives':['pick','place']},states)
        with self.assertRaises(ValueError):validate_plan({'steps':[arm_step('right','place')]},['place'],True,{'primitives':['place']},states)

    def test_right_only_complete_pick(self):
        with tempfile.TemporaryDirectory() as folder:
            robot=DualRobot()
            planner=ScriptedPlanner([[arm_step('right','pick')]],
                [answer(uv=[.4,.4]),answer('step_done'),answer('task_done')])
            self.assertEqual(run(self.setup_task(folder,'right'),Journal(folder),robot,planner),0)
            self.assertTrue(robot.held['right']);self.assertFalse(robot.held['left'])

    def test_two_arm_pick_preserves_both_held_states(self):
        with tempfile.TemporaryDirectory() as folder:
            robot=DualRobot()
            planner=ScriptedPlanner([[arm_step('left','pick'),arm_step('right','pick')]],
                [answer(uv=[.4,.4]),answer('step_done'),answer(uv=[.7,.4]),answer('step_done'),answer('task_done')])
            self.assertEqual(run(self.setup_task(folder),Journal(folder),robot,planner),0)
            self.assertEqual(robot.held,{'left':True,'right':True})

    def test_arm_restriction_rejects_right_plan_before_motor(self):
        with tempfile.TemporaryDirectory() as folder:
            robot=DualRobot();bad=[arm_step('right','pick')]
            planner=ScriptedPlanner([bad,copy.deepcopy(bad)],[])
            with self.assertRaises(ValueError):run(self.setup_task(folder,'left'),Journal(folder),robot,planner)
            self.assertFalse(any(m=='dual.execute' for m,a in robot.calls))

    def test_unknown_receipt_never_switches_arm_and_replays(self):
        with tempfile.TemporaryDirectory() as folder:
            robot=DualRobot(lost=True)
            planner=ScriptedPlanner([[arm_step('left','pick'),arm_step('right','pick')]],[answer(uv=[.5,.5])])
            with self.assertRaises(TimeoutError):run(self.setup_task(folder),Journal(folder),robot,planner)
            self.assertEqual(sum(m=='dual.execute' for m,a in robot.calls),1)

    def test_occlusion_can_clear_other_arm_before_retrying(self):
        with tempfile.TemporaryDirectory() as folder:
            robot=DualRobot()
            blocked=dict(check('blocked',reason='occluded'),code='')
            planner=ScriptedPlanner([[arm_step('left','pick')],
                [arm_step('right','clear_view'),arm_step('left','pick')]],
                [blocked,answer(),answer('step_done'),answer(uv=[.5,.5]),answer('step_done'),answer('task_done')])
            log=Journal(folder)
            self.assertEqual(run(self.setup_task(folder),log,robot,planner),0)
            plans=[a for m,a in robot.calls if m=='dual.prepare']
            self.assertEqual([(a[0],a[2]) for a in plans],[('right','clear_view'),('left','pick')])

    def test_mismatched_arm_receipt_is_unknown_not_success(self):
        class WrongReceipt(DualRobot):
            def call(self,method,*args,**kwargs):
                result=super().call(method,*args,**kwargs)
                if method=='dual.execute':result['selected_arm']='right'
                return result
        with tempfile.TemporaryDirectory() as folder:
            robot=WrongReceipt()
            planner=ScriptedPlanner([[arm_step('left','pick')]],[answer(uv=[.5,.5])])
            with self.assertRaisesRegex(RuntimeError,'another arm'):
                run(self.setup_task(folder),Journal(folder),robot,planner)
            self.assertEqual(sum(m=='dual.execute' for m,a in robot.calls),1)

    def test_dual_motion_only_task_can_finish(self):
        with tempfile.TemporaryDirectory() as folder:
            robot=DualRobot()
            planner=ScriptedPlanner([[arm_step('left','retreat'),arm_step('right','retreat')]],
                [answer(),answer('step_done'),answer(),answer('step_done'),answer('task_done')])
            self.assertEqual(run(self.setup_task(folder),Journal(folder),robot,planner),0)

    def test_final_verification_detects_drop_on_previously_moved_arm(self):
        from cix_harness.executive import TaskExecutive
        executive=object.__new__(TaskExecutive)
        executive.verified=[dict(arm_step('left','retreat'),verified_gripping=True),
                            dict(arm_step('right','retreat'),verified_gripping=False)]
        self.assertFalse(executive.final_state_consistent({'arm_states':{
            'left':{'gripping':False},'right':{'gripping':False}}}))
