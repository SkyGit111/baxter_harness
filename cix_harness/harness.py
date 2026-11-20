"""Task planning, skill composition, verification and memory over either existing executor."""
import argparse
from pathlib import Path

from .__main__ import Journal, run as run_vla
from .api_planner import DEFAULT_API_MODEL, ResponsesTaskPlanner
from .contract import instruction
from .executive import ExecutiveJournal, TaskExecutive
from .memory import EpisodeMemory, scope_key
from .primitive_runner import run as run_primitives
from .rpc import HttpRpcClient
from .task_plan import VisualTaskPlanner
from .timing import timed_rpc


def run(args, journal, robot=None, planner=None):
    if getattr(args,'planner_provider','local')=='api' and getattr(args,'planner_view','scene')!='scene':
        raise ValueError('API planning requires D455 as the primary view; use --api-images scene or scene_wrist')
    robot = timed_rpc(robot or HttpRpcClient(args.robot_port),journal)
    health = robot.call('healthz')
    arms=getattr(args,'arms','auto')
    if health.get('dual_protocol') or arms!='auto':
        from .arm_client import ArmClient
        # Adapt below the single timing wrapper, so each transport request is
        # counted once rather than as both primitive.* and dual.* round trips.
        robot=ArmClient(robot.client,health,arms)
        if robot.dual:
            if args.route=='vla':raise ValueError('Dual bridge currently exposes primitives; use hybrid for arm selection')
            if args.route=='primitives':
                import copy
                args=copy.copy(args);args.route='hybrid'
            journal('arm_mode',available_arms=robot.allowed,execution='serialized')
        robot=timed_rpc(robot,journal)
        health=robot.call('healthz')
    if args.execute and health.get('armed') is not True: raise RuntimeError('Robot is read-only')
    if args.route == 'primitives' and health.get('motion_scope') != 'supervised_primitive_task':
        raise RuntimeError('Primitive route requires the primitive bridge profile')
    if args.route == 'vla' and args.execute and health.get('motion_scope') not in ('supervised_policy_task', 'task_workspace'):
        raise RuntimeError('Managed VLA tasks require a task bridge, not a diagnostic/primitive profile')
    provider=getattr(args,'planner_provider','local')
    if planner is None and provider=='api':
        planner=ResponsesTaskPlanner(args.api_model,args.api_base_url,args.api_key_env,args.api_timeout,journal,
            auth=getattr(args,'api_auth','api-key'),credential_path=getattr(args,'chatgpt_credentials',None),
            image_mode=getattr(args,'api_images','scene_wrist'))
    planner=planner or VisualTaskPlanner(args.vlm_port,args.vlm_model)
    identity=(getattr(args,'api_base_url','')+':'+getattr(args,'api_model','')) if provider=='api' else args.vlm_model
    memory = EpisodeMemory(args.memory)
    executive = None
    try:
        executive = TaskExecutive(instruction(args.task), args.route,
            planner, memory, scope_key(health, args.route, identity), journal,
            mode='live' if args.execute else 'preview', max_replans=args.max_replans,
            max_calls=args.max_planner_calls, snapshot=Path(journal.folder)/'task-state.json')
        log = ExecutiveJournal(journal, executive)
        if args.route=='hybrid':
            from .hybrid import run as run_hybrid
            from .code_tool import CodeTool
            result=run_hybrid(args,log,executive,robot=robot,code_tool=CodeTool())
        elif args.route == 'primitives':
            result = run_primitives(args, log, robot=robot,
                planner=lambda port, model, goal, obs, history: executive.choose(obs),
                recovery_allowed=executive.recovery_allowed)
        else:
            args.image = None; args.managed_harness = True
            result = run_vla(args, log,
                planner=lambda port, goal, obs, history, model: executive.choose(obs))
        if not executive.closed: executive.finish('unknown' if args.execute else 'preview')
        return result
    except BaseException:
        if executive: executive.finish('unknown')
        raise
    finally:
        memory.close()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True)
    p.add_argument('--route', choices=['primitives', 'vla','hybrid'], default='primitives')
    p.add_argument('--arms',choices=['auto','left','right','both'],default='auto')
    p.add_argument('--planner-provider',choices=['local','api'],default='local')
    p.add_argument('--api-model',default=DEFAULT_API_MODEL,help='Responses planner model (default: %(default)s)')
    p.add_argument('--api-base-url',default='https://api.openai.com/v1')
    p.add_argument('--api-key-env',default='OPENAI_API_KEY')
    p.add_argument('--api-auth',choices=['api-key','chatgpt'],default='api-key')
    p.add_argument('--chatgpt-credentials',type=Path,help='This project\'s OAuth store; defaults to user configuration directory')
    p.add_argument('--api-timeout',type=float,default=120)
    p.add_argument('--api-images',choices=['scene','scene_wrist'],default='scene_wrist',
                   help='D455 only or D455 plus a fresh selected-arm wrist view; uv always uses D455')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--vlm-port', type=int, default=9110)
    p.add_argument('--vlm-model', default='qwen35-9b')
    p.add_argument('--vla-port', type=int, default=9101)
    p.add_argument('--robot-port', type=int, default=9102)
    p.add_argument('--task-kind', choices=['generic', 'pick', 'place', 'stack'], default='generic')
    p.add_argument('--max-turns', type=int, choices=range(1, 33), default=8)
    p.add_argument('--max-replans', type=int, choices=range(0, 5), default=2)
    p.add_argument('--max-planner-calls', type=int, choices=range(2, 65), default=32)
    p.add_argument('--max-recoveries', type=int, choices=range(0, 9), default=3)
    p.add_argument('--max-grasp-retries', type=int, choices=range(0, 4), default=1)
    p.add_argument('--steps', type=int, choices=range(1, 65), default=3)
    p.add_argument('--action-window', type=int, choices=range(1, 17), default=1)
    p.add_argument('--execution-mode', choices=['waypoints', 'segment'], default='waypoints')
    p.add_argument('--planner-view', choices=['scene', 'wrist'], default='scene')
    p.add_argument('--memory', default='.local/harness-memory.sqlite3')
    p.add_argument('--logs', default='runs')
    args = p.parse_args(argv); log = Journal(args.logs)
    try: return run(args, log)
    except (Exception, KeyboardInterrupt) as error:
        log('stop', reason=str(error) or type(error).__name__, replayed=False)
        return 1


if __name__ == '__main__': raise SystemExit(main())
