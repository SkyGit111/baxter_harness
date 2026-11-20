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
from cix_harness.api_planner import DEFAULT_API_MODEL, ResponsesTaskPlanner
from cix_harness.contract import image_data
from cix_harness.planner_images import select_images
from cix_harness.task_plan import compact_state,validate_check


def read_events(folder):
    return [json.loads(line) for line in (Path(folder)/'events.jsonl').read_text(encoding='utf-8').splitlines()]


def frame_index(events):
    """Resolve checkpoint calls within their recorded observation, also on reused frames."""
    frames = []
    for event in events:
        kind = event['event']
        if kind == 'observation':
            frames.append({'frame': event, 'inputs': [], 'replies': [], 'plan': None})
        elif frames:
            item = frames[-1]
            if kind == 'task_plan' and item['plan'] is None:
                item['plan'] = event
            if event.get('obs_id') != item['frame']['obs_id']:
                continue
            if kind == 'checkpoint_input':
                item['inputs'].append(event)
            elif kind == 'task_model_reply' and event.get('phase') == 'check':
                item['replies'].append(event)
    return frames


def list_frames(events):
    return [{'frame': index, 'obs_id': item['frame']['obs_id'], 'file': item['frame']['file'],
             'arm': item['frame']['state'].get('arm'), 'gripping': item['frame']['state'].get('gripping'),
             'saved_checkpoint_inputs': len(item['inputs']),
             'recorded_decisions': [entry['reply'].get('decision') for entry in item['replies']]}
            for index, item in enumerate(frame_index(events))]


def load_run(folder, index=0, checkpoint=0):
    folder = Path(folder).resolve()
    events = read_events(folder)
    frames = frame_index(events)
    if not 0 <= index < len(frames):
        raise ValueError('Frame index is outside the saved run; use --list')
    item = frames[index]
    frame = item['frame']
    source = (folder/frame['file']).resolve()
    if source.parent != folder:
        raise ValueError('Image must belong to the selected run')
    if frame.get('role') == 'planner-wrist' or source.stem.endswith('-planner-wrist'):
        raise ValueError('This historical frame used the wrist as primary; D455 is unavailable')
    obs = {'id': frame['obs_id'], 'state': frame['state'], 'image': image_data(source.read_bytes()),
           'camera_metadata': frame.get('camera_metadata', {})}
    wrist = source.with_name(source.stem+'-wrist.jpg').resolve()
    if wrist.parent != folder:
        raise ValueError('Wrist image must belong to the selected run')
    if wrist.exists():
        obs['wrist_image'] = image_data(wrist.read_bytes())
    if not 0 <= checkpoint < max(len(item['inputs']), len(item['replies']), 1):
        raise ValueError('Checkpoint index is outside the selected frame; use --list')
    context = None
    origin = 'missing'
    if item['inputs']:
        entry = item['inputs'][checkpoint]
        for key in ('workspace', 'arm_states', 'available_arms', 'workspaces', 'arm_skills',
                    'arm_recovery', 'camera_metadata', 'backend_skills'):
            if key in entry['observation_metadata']:
                obs[key] = entry['observation_metadata'][key]
        context = entry['context']
        origin = 'recorded_checkpoint'
    elif index == 0 and checkpoint == 0 and item['plan'] and item['plan']['steps']:
        # Compatibility with the original initial-frame approximation. No later
        # executed/verified state is inferred when the input was never recorded.
        start = next(e for e in events if e['event'] == 'task_started')
        begin = next((e for e in events if e['event'] == 'start'), {})
        if 'backend_skills' in begin:
            obs['backend_skills'] = begin['backend_skills']
        context = {'goal': start['goal'], 'route': start['route'], 'state': compact_state(obs),
                   'current': item['plan']['steps'][0], 'verified': [],
                   'comparison_note': 'Frozen historical observation with reconstructed initial context.'}
        origin = 'legacy_initial_reconstruction'
    obs['_replay'] = {'frame': index, 'checkpoint': checkpoint, 'context_source': origin}
    return obs, context


def compare(obs,context,*,call_api=False,model=DEFAULT_API_MODEL,auth='api-key',credential_path=None,repeats=1):
    manifest={mode:select_images(obs,mode)[1] for mode in ('scene','scene_wrist')}
    result={'mode':'saved_observation_readonly','robot_calls':0,'api_requested':call_api,
            'obs_id':obs.get('id'),'selection':obs.get('_replay'),'images':manifest,'context':context,'trials':[],
            'limitations':['Frozen-frame decision/latency comparison; no physical success measurement.',
                           'One example cannot establish a success-rate improvement.',
                           'Capture age is from historical observation assembly, not current wall time.']}
    if not call_api:return result
    if manifest['scene_wrist']['image_count']!=2:
        raise ValueError('Cannot compare two-view input: '+manifest['scene_wrist']['wrist_omitted_reason'])
    if context is None:
        raise ValueError('This frame has no saved checkpoint input. An offline image manifest is available.')
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
    p.add_argument('--list',action='store_true',help='List saved frame indices, arm and checkpoint decisions')
    p.add_argument('--frame',type=int,default=0,help='Zero-based observation index from --list')
    p.add_argument('--checkpoint',type=int,default=0,help='Zero-based checkpoint call within this frame')
    p.add_argument('--output',type=Path,help='New local JSON result file')
    p.add_argument('--call-api',action='store_true')
    p.add_argument('--model',default=DEFAULT_API_MODEL,help='Responses planner model (default: %(default)s)')
    p.add_argument('--auth',choices=['api-key','chatgpt'],default='api-key')
    p.add_argument('--credentials',type=Path)
    p.add_argument('--repeats',type=int,choices=range(1,4),default=1)
    args=p.parse_args(argv)
    if args.list:
        print(json.dumps(list_frames(read_events(args.run)),indent=2,ensure_ascii=False))
        return 0
    if args.output is None:p.error('--output is required for comparison')
    obs,context=load_run(args.run,args.frame,args.checkpoint)
    if args.call_api and context is None:
        p.error('This frame lacks its checkpoint context; omit --call-api for an offline manifest')
    # Reserve the result before spending tokens; never overwrite prior evidence.
    with args.output.open('x',encoding='utf-8') as output:
        result=compare(obs,context,call_api=args.call_api,model=args.model,auth=args.auth,
                       credential_path=args.credentials,repeats=args.repeats)
        json.dump(result,output,indent=2,ensure_ascii=False)
    print(json.dumps({'output':str(args.output),'robot_calls':0,'api_trials':len(result['trials']),
                      'input_views':result['images']},ensure_ascii=False))
    return 0


if __name__=='__main__':raise SystemExit(main())
