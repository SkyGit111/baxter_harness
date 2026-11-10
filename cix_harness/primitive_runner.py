"""VLM + deterministic primitive runner. No VLA service is contacted."""
import argparse
import json
from .contract import instruction, vector
from .primitives import SKILLS, number
from .rpc import HttpRpcClient, local_json

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
        'recovery':obs.get('recovery',{}),'recent':history[-3:]},ensure_ascii=False,separators=(',',':'))},
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


def run(args,log,robot=None,planner=decide,recovery_allowed=None):
    robot=robot or HttpRpcClient(args.robot_port)
    status=robot.call('healthz')
    if status.get('motion_scope')!='supervised_primitive_task':raise RuntimeError('Start the primitive bridge profile')
    if args.execute and status.get('armed') is not True:raise RuntimeError('Robot is read-only')
    history=[];completed=[];used_views=set();recoveries=0;actions=0;empty_grasps=0;retract_pending=False
    max_recoveries=getattr(args,'max_recoveries',3)
    max_grasp_retries=getattr(args,'max_grasp_retries',1)
    log('start',route='primitives',goal=instruction(args.task),mode='live' if args.execute else 'preview')
    # A preflight rejection spends recovery budget, but no motion/skill turn.
    for _ in range(args.max_turns+max_recoveries+1):
        obs=robot.call('primitive.observe');log.camera(obs,'primitive')
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
            log('finish',status='done' if okay else 'blocked_or_unverified',evidence=decision['evidence'])
            return 0 if okay else 2
        if actions>=args.max_turns:log('finish',status='turn_budget_exhausted');return 2
        options={'kwargs':{'view_index':view_index}} if view_index is not None else {}
        if view_index is not None:used_views.add(view_index)
        prepared=robot.call('primitive.prepare',(obs['id'],decision['skill'],decision['uv']),timeout_s=60,**options)
        log('primitive_plan',**prepared)
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
        if receipt.get('applied') is not True or receipt.get('plan_id')!=prepared['plan_id']:
            raise RuntimeError('Invalid primitive acknowledgement')
        log('primitive_result',**receipt);actions+=1
        history.append(feedback(decision,receipt))
        if receipt.get('status','completed')=='failed':
            if (decision['skill']!='pick' or receipt.get('failure_code')!='empty_grasp'
                or receipt.get('recoverable') is not True or receipt.get('stopped') is not True
                or receipt.get('gripping') is not False):
                raise RuntimeError('Primitive failure is not safe to recover automatically')
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
