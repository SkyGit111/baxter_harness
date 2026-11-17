"""Compare D455 vs D455+wrist on a saved frame; no robot client or motor calls.

Default is an offline manifest only. --call-api explicitly makes two (or more)
read-only API requests. This compares decisions/latency, not physical success.
"""
import argparse
import copy
import json
from pathlib import Path
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from cix_harness.api_planner import ResponsesTaskPlanner
from cix_harness.contract import image_data
from cix_harness.planner_images import select_images
from cix_harness.task_plan import compact_state,validate_check


def load_run(folder,index=0):
    folder=Path(folder)
    events=[json.loads(line) for line in (folder/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    frames=[e for e in events if e.get('event')=='observation']
    frame=frames[index]
    source=folder/frame['file']
    if source.parent.resolve()!=folder.resolve():raise ValueError('Image must belong to the selected run')
    obs={'id':frame['obs_id'],'state':frame['state'],'image':image_data(source.read_bytes())}
    wrist=source.with_name(source.stem+'-wrist.jpg')
    if wrist.exists():obs['wrist_image']=image_data(wrist.read_bytes())
    start=next(e for e in events if e.get('event')=='task_started')
    plans=[e for e in events if e.get('event')=='task_plan']
    if not plans:raise ValueError('Saved run has no semantic plan for a matched checkpoint comparison')
    # An initial observation/checkpoint is reproducible without inventing state
    # after motion. Later observations require their actual saved context.
    if index!=0:raise ValueError('Run replay currently supports the initial observation only')
    current=copy.deepcopy(plans[0]['steps'][0])
    begin=next((e for e in events if e.get('event')=='start'),{})
    if 'backend_skills' in begin:obs['backend_skills']=begin['backend_skills']
    context={'goal':start['goal'],'route':start['route'],'current':current,'state':compact_state(obs),
             'verified':[],'comparison_note':'Frozen historical observation; no live robot action will follow.'}
    return obs,context


def compare(obs,context,*,call_api=False,model='gpt-6-astra',auth='api-key',credential_path=None,repeats=1):
    manifest={mode:select_images(obs,mode)[1] for mode in ('scene','scene_wrist')}
    result={'mode':'saved_observation_readonly','robot_calls':0,'api_requested':call_api,
            'obs_id':obs.get('id'),'images':manifest,'context':context,'trials':[],
            'limitations':['Frozen-frame decision/latency comparison; no physical success measurement.',
                           'One example cannot establish a success-rate improvement.',
                           'Capture age is from historical observation assembly, not current wall time.']}
    if not call_api:return result
    if manifest['scene_wrist']['image_count']!=2:
        raise ValueError('Cannot compare two-view input: '+manifest['scene_wrist']['wrist_omitted_reason'])
    for repeat in range(repeats):
        # Reverse repeated order to reduce a fixed first-call bias.
        for mode in (('scene','scene_wrist') if repeat%2==0 else ('scene_wrist','scene')):
            events=[]
            def log(event,**fields):events.append(dict(event=event,**fields))
            planner=ResponsesTaskPlanner(model=model,auth=auth,credential_path=credential_path,
                                         image_mode=mode,log=log)
            trial={'image_mode':mode,'repeat':repeat,'events':events}
            try:
                reply=planner.check(copy.deepcopy(obs),copy.deepcopy(context))
                validate_check(reply,context['route'],context['current'])
                trial.update(status='completed',decision=reply)
            except Exception as error:
                # Exception text can contain provider/account content.
                trial.update(status='error',error_type=type(error).__name__)
            result['trials'].append(trial)
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--call-api',action='store_true')
    p.add_argument('--model',default='gpt-6-astra')
    p.add_argument('--auth',choices=['api-key','chatgpt'],default='api-key')
    p.add_argument('--credentials',type=Path)
    p.add_argument('--repeats',type=int,choices=range(1,4),default=1)
    args=p.parse_args(argv)
    obs,context=load_run(args.run)
    # Reserve the result before spending tokens; never overwrite prior evidence.
    with args.output.open('x',encoding='utf-8') as output:
        result=compare(obs,context,call_api=args.call_api,model=args.model,auth=args.auth,
                       credential_path=args.credentials,repeats=args.repeats)
        json.dump(result,output,indent=2,ensure_ascii=False)
    print(json.dumps({'output':str(args.output),'robot_calls':0,'api_trials':len(result['trials']),
                      'input_views':result['images']},ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
