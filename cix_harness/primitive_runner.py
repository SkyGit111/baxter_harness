"""VLM + deterministic primitive runner. No VLA service is contacted."""
import argparse
import json
from .contract import instruction, vector
from .primitives import SKILLS, number
from .rpc import HttpRpcClient, local_json
from . import ring_recheck

SCHEMA={'type':'object','properties':{
 'decision':{'type':'string','enum':['act','done','blocked']},
 'reason':{'type':'string','enum':['none','occluded','ambiguous_target','not_found','unsafe','unreachable']},
 'skill':{'type':'string','enum':['',*SKILLS]},
 'uv':{'anyOf':[{'type':'null'},{'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2}]},
 'evidence':{'type':'string','minLength':1,'maxLength':300}},'required':['decision','reason','skill','uv','evidence'],'additionalProperties':False}
SYSTEM='''Control state.arm using the CURRENT image, telemetry and available_skills.
Follow the Chinese/English goal. Robot-left may appear image-right; do not confuse arms.
pick handles approach, orientation, opening, closing, grasp verification and lift;
being far from the object or empty/closed is not a reason to block it. IK checks reachability.
place/stack transport, release onto a support and retreat. Never pick while holding;
never open in mid-air. gripping=false means empty even if closed; percent 0=closed,100=open.
Pick uv on the object's visible TOP interior; place uv on a SUPPORT surface;
stack uv on the receiving block's TOP. move_to has no height offset.
uv is normalized: top-left [0,0], bottom-right [1,1]. Other skills use null.
Never invent 3D coordinates or infer a pixel behind an obstruction.
If an arm hides the target/support, return blocked, reason=occluded. The harness can
clear_view (move aside without changing grip), capture a NEW image and ask again.
Other blocked reasons: ambiguous_target (identity unclear), not_found, unsafe, unreachable.
Do not reuse old pixels. target_depth failure: select another interior point away from
edges/holes. ik_preflight failure: reconsider target, not blind repetition.
After empty_grasp the harness retracts; relocalize in the new image before retrying.
done requires visual outcome AND telemetry. A pick must hold and lift the object;
clear_view or moving an empty gripper alone cannot complete it.
Return decision,reason,skill,uv,evidence JSON only. act/done reason=none;
done/blocked skill="", uv=null. Evidence at most 120 characters.'''

def decide(port,model,goal,obs,history):
    state={k:obs['state'][k] for k in ('arm','eef_m','eef_xyzw','gripping','gripper_percent') if k in obs['state']}
    for key in ('eef_m','eef_xyzw'):
        if key in state:state[key]=[round(v,4) for v in state[key]]
    body={'model':model,'temperature':0,'max_tokens':256,'messages':[
      {'role':'system','content':SYSTEM}, {'role':'user','content':[
      {'type':'text','text':json.dumps({'goal':goal,'state':state,
        'available_skills':obs.get('primitive_skills',[s for s in SKILLS if s!='clear_view']),
        'recovery':obs.get('recovery',{}),'recent':history[-3:],
        **({'ring_scene':obs['ring_scene'],'ring_task':'Pick loose rings only. Select the matching marked peg by its tip. When held_transport is configured and the task requires placement, use move_to on that tip until at_hover, then place. Verify using verified_skill.'}
           if 'ring_scene' in obs else {})},ensure_ascii=False,separators=(',',':'))},
      {'type':'image_url','image_url':{'url':obs['image']}}]}],
      'response_format':{'type':'json_schema','json_schema':{'name':'primitive_decision','strict':True,'schema':SCHEMA}}}
    choice=local_json(port,'/v1/chat/completions',body,timeout=180)['choices'][0]
    if choice.get('finish_reason')=='length':raise ValueError('Truncated primitive decision')
    d=json.loads(choice['message']['content'])
    return validate_decision(d)


def validate_decision(d):
    if not isinstance(d,dict) or set(d)!=set(SCHEMA['required']):raise ValueError('Invalid primitive decision keys')
    instruction(d['evidence'])
    if len(d['evidence'])>300:raise ValueError('Primitive evidence too long')
    if d['reason'] not in SCHEMA['properties']['reason']['enum']:raise ValueError('Invalid decision reason')
    if (d['decision']=='blocked') != (d['reason']!='none'):raise ValueError('Reason must explain a blocked decision')
    if d['decision']=='act':
        if d['skill'] not in SKILLS:raise ValueError('Unknown skill')
        if d['skill'] in ('pick','place','stack','move_to'):
            for v in vector(d['uv'],2,'target uv'):number(v,0,1,'pixel coordinate')
        elif d['uv'] is not None:raise ValueError('Skill does not accept a pixel target')
    elif d['decision'] not in ('done','blocked') or d['skill']!='' or d['uv'] is not None:
        raise ValueError('Invalid terminal decision')
    return d


def feedback(decision,receipt):
    """Keep motor paths and full stage receipts in the journal, not the VLM context."""
    stages=receipt.get('completed_stages',[])
    return {'skill':decision['skill'],'uv':decision['uv'],
            'status':receipt.get('status','completed'),'failure_code':receipt.get('failure_code'),
            'message':receipt.get('message','')[:200],
            'gripping':receipt.get('gripping'),'motion_remaining':receipt.get('motion_remaining'),
            'eef_m':stages[-1].get('eef_m') if stages else None}


def action(skill,evidence):
    return {'decision':'act','reason':'none','skill':skill,'uv':None,'evidence':evidence}


def log_execution_receipt(log, receipt, plan_id):
    if not isinstance(receipt,dict) or receipt.get('plan_id')!=plan_id:
        raise RuntimeError('Invalid primitive acknowledgement')
    if receipt.get('status')=='failed':
        if 'applied' not in receipt or not any(receipt['applied'] is value for value in (True,False,None)):
            raise RuntimeError('Invalid failed primitive acknowledgement')
    elif receipt.get('applied') is not True:
        raise RuntimeError('Invalid primitive acknowledgement')
    log('primitive_result',**receipt)


def finish_execution_failure(log,receipt):
    uncertain=receipt.get('outcome_uncertain') is not False or (receipt.get('stop_attempted') is True and receipt.get('stopped') is not True)
    log('finish',status=receipt.get('failure_code','execution_failed'),plan_id=receipt['plan_id'],
        evidence=receipt.get('message',''),outcome_uncertain=uncertain,replayed=False)
    return 2


def peer_moved_before_dispatch(receipt):
    """Recognize a consumed plan refused before its first command, including older bridges."""
    stage=receipt.get('failed_stage') or {}
    reason=receipt.get('message') or ''
    legacy=(receipt.get('failure_type')=='PeerStateChangedError' and
            (reason=='Inactive arm moved since observation; reobserve before motion' or
             reason.startswith('Inactive arm moved since observation: ')))
    return (receipt.get('status')=='failed' and receipt.get('failure_code')=='execution_failed' and
            (receipt.get('failure_type')=='PeerJointMovedError' or legacy) and
            receipt.get('applied') is False and receipt.get('outcome_uncertain') is False and
            type(receipt.get('confirmed_commands')) is int and receipt['confirmed_commands']==0 and
            type(receipt.get('completed_stage_count')) is int and receipt['completed_stage_count']==0 and
            receipt.get('completed_stages')==[] and isinstance(stage,dict) and
            type(stage.get('stage_index')) is int and stage['stage_index']==0 and
            stage.get('status')=='failed' and stage.get('command_state')=='not_dispatched' and
            receipt.get('stop_attempted') is False and receipt.get('stopped') is None and
            not receipt.get('stop_error') and not receipt.get('arm_state_errors'))


class PeerMotionRecovery:
    """One fresh observation and new planning opportunity per task. Never reuse the old token."""
    cause='peer_moved_before_dispatch'

    def __init__(self):
        self.plan_id=None
        self.waiting_for_observation=False
        self.waiting_for_plan=False

    def request(self,receipt,obs,remaining,allowed,log):
        if self.plan_id is not None or remaining<=0 or not peer_moved_before_dispatch(receipt):
            return False
        state=obs['state']
        if type(state.get('gripping')) is not bool or (allowed and not allowed(self.cause)):
            return False
        self.plan_id=receipt['plan_id'];self.obs_id=obs['id']
        self.arm=state.get('arm');self.gripping=state['gripping']
        self.waiting_for_observation=True;self.waiting_for_plan=True
        log('recovery',kind='reobserve_peer_motion',cause=self.cause,attempt=1,
            retired_plan_id=self.plan_id,previous_obs_id=self.obs_id,replayed=False)
        return True

    def observation_error(self,obs,log):
        if not self.waiting_for_observation:return None
        state=obs['state']
        if not obs.get('id') or obs['id']==self.obs_id:return 'peer_recovery_observation_not_fresh'
        if state.get('arm')!=self.arm or state.get('gripping') is not self.gripping:
            return 'active_grasp_or_arm_changed_during_peer_recovery'
        self.waiting_for_observation=False
        log('peer_motion_reobserved',obs_id=obs['id'],previous_obs_id=self.obs_id,
            retired_plan_id=self.plan_id,arm=self.arm,gripping=self.gripping)
        return None

    def plan_error(self,prepared,obs,log):
        if self.plan_id is None:return None
        if prepared.get('plan_id')==self.plan_id:return 'peer_recovery_plan_token_reused'
        if not self.waiting_for_plan or prepared.get('status')=='rejected':return None
        if not prepared.get('plan_id') or prepared.get('obs_id',obs['id'])!=obs['id'] or obs['id']==self.obs_id:
            return 'peer_recovery_plan_without_fresh_observation'
        self.waiting_for_plan=False
        log('peer_motion_replanned',obs_id=obs['id'],plan_id=prepared['plan_id'],retired_plan_id=self.plan_id)
        return None


def run(args,log,robot=None,planner=decide,recovery_allowed=None):
    from .timing import timed_rpc
    robot=timed_rpc(robot or HttpRpcClient(args.robot_port),log)
    status=robot.call('healthz')
    if status.get('motion_scope')!='supervised_primitive_task':raise RuntimeError('Start the primitive bridge profile')
    if args.execute and status.get('armed') is not True:raise RuntimeError('Robot is read-only')
    history=[];completed=[];used_views=set();recoveries=0;actions=0;empty_grasps=0;retract_pending=False
    max_recoveries=getattr(args,'max_recoveries',3)
    max_grasp_retries=getattr(args,'max_grasp_retries',1)
    peer_recovery=PeerMotionRecovery()
    log('start',route='primitives',goal=instruction(args.task),mode='live' if args.execute else 'preview')
    def observe():
        obs=robot.call('primitive.observe');log.camera(obs,'primitive');return obs

    # A preflight rejection spends recovery budget, but no motion/skill turn.
    for _ in range(args.max_turns+max_recoveries+1):
        obs=observe()
        problem=peer_recovery.observation_error(obs,log)
        if problem:log('finish',status=problem);return 2
        views=[i for i in obs.get('recovery',{}).get('view_indices',[]) if i not in used_views]
        view_index=None;is_retraction=retract_pending
        if retract_pending:
            if obs['state'].get('gripping') is not False:
                log('finish',status='grasp_state_changed_during_recovery');return 2
            decision=action('clear_view' if views else 'retreat','Retract after confirmed empty grasp before relocalizing')
        else:
            decision=validate_decision(planner(args.vlm_port,args.vlm_model,args.task,obs,history))
        log('decision',**decision)
        if decision['decision']=='blocked' and decision['reason']=='occluded':
            log('recoverable_block',**decision)
            if recovery_allowed and not recovery_allowed('occluded'):
                log('finish',status='recovery_policy_stop');return 2
            decision=action('clear_view','Move aside and acquire a new image of the occluded target')
        if decision['decision']=='act' and decision['skill']=='clear_view':
            if not views:
                log('finish',status='no_unused_observation_pose',evidence='Configure a reachable view-clearing TCP position; target pixels cannot be guessed');return 2
            view_index=views[0]
            if not is_retraction:
                if recoveries>=max_recoveries:log('finish',status='recovery_budget_exhausted');return 2
                recoveries+=1
            log('recovery',kind='clear_view',view_index=view_index,attempt=recoveries)
        if decision['decision']!='act':
            okay=decision['decision']=='done'
            if args.execute and okay:
                if args.task_kind=='pick':okay='pick' in completed and obs['state'].get('gripping') is True
                elif args.task_kind in ('place','stack'):
                    okay=args.task_kind in completed and obs['state'].get('gripping') is False
                if 'ring_scene' in obs and completed:okay=okay and obs['ring_scene'].get('verified_skill')==completed[-1]
            log('finish',status='done' if okay else 'blocked_or_unverified',evidence=decision['evidence'])
            return 0 if okay else 2
        if actions>=args.max_turns:log('finish',status='turn_budget_exhausted');return 2
        options={'kwargs':{'view_index':view_index}} if view_index is not None else {}
        if view_index is not None:used_views.add(view_index)
        prepared,obs,decision,spent=ring_recheck.prepare(robot,log,obs,decision,options,observe,
            max_recoveries-recoveries,recovery_allowed)
        recoveries+=spent
        log('primitive_plan',**prepared)
        problem=peer_recovery.plan_error(prepared,obs,log)
        if problem:log('finish',status=problem);return 2
        if prepared.get('status')=='rejected':
            history.append(feedback(decision,prepared))
            if prepared.get('recoverable') is not True or prepared.get('applied') is not False:
                log('finish',status=prepared.get('failure_code','preflight_rejected'));return 2
            if recoveries>=max_recoveries:log('finish',status='recovery_budget_exhausted');return 2
            recoveries+=1;log('recovery',kind='reobserve',cause=prepared['failure_code'],attempt=recoveries)
            continue
        if not args.execute:return 0
        # A lost response can follow real motion: never automatically retry this call.
        receipt=robot.call('primitive.execute',(prepared['plan_id'],),timeout_s=600)
        log_execution_receipt(log,receipt,prepared['plan_id'])
        history.append(feedback(decision,receipt))
        if peer_recovery.request(receipt,obs,max_recoveries-recoveries,recovery_allowed,log):
            recoveries+=1
            if view_index is not None:used_views.discard(view_index)
            continue
        actions+=1
        if receipt.get('status','completed')=='failed':
            if (decision['skill']!='pick' or receipt.get('failure_code')!='empty_grasp'
                or receipt.get('recoverable') is not True or receipt.get('stopped') is not True
                or receipt.get('gripping') is not False or receipt.get('applied') is not True):
                return finish_execution_failure(log,receipt)
            if empty_grasps>=max_grasp_retries or recoveries>=max_recoveries:
                log('finish',status='grasp_recovery_budget_exhausted');return 2
            if recovery_allowed and not recovery_allowed('empty_grasp'):
                log('finish',status='recovery_policy_stop');return 2
            empty_grasps+=1;recoveries+=1;retract_pending=True;used_views.clear()
            log('recovery',kind='empty_grasp_retraction',attempt=empty_grasps)
            continue
        if receipt.get('status','completed')!='completed':raise RuntimeError('Unknown primitive outcome')
        if is_retraction:retract_pending=False
        elif decision['skill']!='clear_view':
            completed.append(decision['skill']);used_views.clear()
    return 2

def main():
    from .__main__ import Journal
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--vlm-port',type=int,default=9110);p.add_argument('--vlm-model',default='qwen35-9b')
    p.add_argument('--robot-port',type=int,default=9102);p.add_argument('--max-turns',type=int,default=8,choices=range(1,33))
    p.add_argument('--max-recoveries',type=int,default=3,choices=range(0,9))
    p.add_argument('--max-grasp-retries',type=int,default=1,choices=range(0,4))
    p.add_argument('--task-kind',choices=['generic','pick','place','stack'],default='pick')
    p.add_argument('--logs',default='runs');args=p.parse_args();log=Journal(args.logs)
    try:return run(args,log)
    except Exception as e:log('error',message=str(e),replayed=False);return 1

if __name__=='__main__':raise SystemExit(main())
