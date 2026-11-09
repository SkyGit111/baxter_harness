"""VLM + deterministic primitive runner. No VLA service is contacted."""
import argparse
import json
from .contract import instruction, vector
from .primitives import SKILLS, number
from .rpc import HttpRpcClient, local_json

SCHEMA={'type':'object','properties':{
 'decision':{'type':'string','enum':['act','done','blocked']},
 'skill':{'type':'string','enum':['',*SKILLS]},
 'uv':{'anyOf':[{'type':'null'},{'type':'array','items':{'type':'number'},'minItems':2,'maxItems':2}]},
 'evidence':{'type':'string'}},'required':['decision','skill','uv','evidence'],'additionalProperties':False}
SYSTEM='''Control one Baxter arm using a fixed deterministic primitive library.
Read the user's Chinese or English goal and the CURRENT image and measured state.
Choose ONE skill: pick, place, stack, move_to, lift, retreat, open, close, home.
For pick identify an interior point on the target object's TOP surface.
For place identify the visible SUPPORT surface (table/container bottom); for stack identify
an interior point on the TOP surface of the receiving block. uv=[u,v] is normalized image
coordinates: top-left [0,0], bottom-right [1,1]. Never invent 3D coordinates or camera calibration.
For move_to select a visible surface target; the controller adds no height offset for this skill.
Other skills use uv=null. A pick includes approach, close, verified grasp and lift.
Place/stack include transport, release and retreat. Never pick while already holding.
Do not open while transporting an object. After each skill inspect the new image and receipt.
A closed empty gripper is not holding: gripping=false. gripper_percent 0=closed,100=open.
Only choose done if the requested physical outcome is visible AND consistent with telemetry.
If the intended target or support surface is occluded/ambiguous choose blocked.
Terminal decisions require skill="" and uv=null. Return only decision,skill,uv,evidence JSON.'''

def decide(port,model,goal,obs,history):
    body={'model':model,'temperature':0,'max_tokens':350,'messages':[
      {'role':'system','content':SYSTEM}, {'role':'user','content':[
      {'type':'text','text':json.dumps({'goal':goal,'state':obs['state'],'recent':history[-4:]},ensure_ascii=False)},
      {'type':'image_url','image_url':{'url':obs['image']}}]}],
      'response_format':{'type':'json_schema','json_schema':{'name':'primitive_decision','strict':True,'schema':SCHEMA}}}
    choice=local_json(port,'/v1/chat/completions',body,timeout=180)['choices'][0]
    if choice.get('finish_reason')=='length':raise ValueError('Truncated primitive decision')
    d=json.loads(choice['message']['content'])
    if not isinstance(d,dict) or set(d)!=set(SCHEMA['required']):raise ValueError('Invalid primitive decision keys')
    instruction(d['evidence'])
    if d['decision']=='act':
        if d['skill'] not in SKILLS:raise ValueError('Unknown skill')
        if d['skill'] in ('pick','place','stack','move_to'):
            for v in vector(d['uv'],2,'target uv'):number(v,0,1,'pixel coordinate')
        elif d['uv'] is not None:raise ValueError('Skill does not accept a pixel target')
    elif d['decision'] not in ('done','blocked') or d['skill']!='' or d['uv'] is not None:
        raise ValueError('Invalid terminal decision')
    return d

def run(args,log,robot=None,planner=decide):
    robot=robot or HttpRpcClient(args.robot_port)
    status=robot.call('healthz')
    if status.get('motion_scope')!='supervised_primitive_task':raise RuntimeError('Start the primitive bridge profile')
    if args.execute and status.get('armed') is not True:raise RuntimeError('Robot is read-only')
    history=[];completed=[]
    log('start',route='primitives',goal=instruction(args.task),mode='live' if args.execute else 'preview')
    for turn in range(args.max_turns+1):
        obs=robot.call('primitive.observe');log.camera(obs,'primitive')
        decision=planner(args.vlm_port,args.vlm_model,args.task,obs,history);log('decision',**decision)
        if decision['decision']!='act':
            okay=decision['decision']=='done'
            if args.execute and okay:
                if args.task_kind=='pick':okay='pick' in completed and obs['state'].get('gripping') is True
                elif args.task_kind in ('place','stack'):
                    okay=args.task_kind in completed and obs['state'].get('gripping') is False
            log('finish',status='done' if okay else 'blocked_or_unverified',evidence=decision['evidence'])
            return 0 if okay else 2
        if turn==args.max_turns:log('finish',status='turn_budget_exhausted');return 2
        prepared=robot.call('primitive.prepare',(obs['id'],decision['skill'],decision['uv']),timeout_s=60)
        log('primitive_plan',**prepared)
        if not args.execute:return 0
        # A lost response can follow real motion: never automatically retry this call.
        receipt=robot.call('primitive.execute',(prepared['plan_id'],),timeout_s=600)
        if receipt.get('applied') is not True or receipt.get('plan_id')!=prepared['plan_id']:
            raise RuntimeError('Invalid primitive acknowledgement')
        log('primitive_result',**receipt);completed.append(decision['skill'])
        history.append({'decision':decision,'receipt':receipt})
    return 2

def main():
    from .__main__ import Journal
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True);p.add_argument('--execute',action='store_true')
    p.add_argument('--vlm-port',type=int,default=9110);p.add_argument('--vlm-model',default='qwen35-9b')
    p.add_argument('--robot-port',type=int,default=9102);p.add_argument('--max-turns',type=int,default=8,choices=range(1,33))
    p.add_argument('--task-kind',choices=['generic','pick','place','stack'],default='pick')
    p.add_argument('--logs',default='runs');args=p.parse_args();log=Journal(args.logs)
    try:return run(args,log)
    except Exception as e:log('error',message=str(e),replayed=False);return 1

if __name__=='__main__':raise SystemExit(main())
