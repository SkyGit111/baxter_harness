"""Task planning, skill composition, verification and memory over either existing executor."""
import argparse
from pathlib import Path

from .__main__ import Journal, run as run_vla
from .contract import instruction
from .executive import ExecutiveJournal, TaskExecutive
from .memory import EpisodeMemory, scope_key
from .primitive_runner import run as run_primitives
from .rpc import HttpRpcClient
from .task_plan import VisualTaskPlanner


def run(args, journal, robot=None, planner=None):
    robot = robot or HttpRpcClient(args.robot_port)
    health = robot.call('healthz')
    if args.execute and health.get('armed') is not True: raise RuntimeError('Robot is read-only')
    if args.route == 'primitives' and health.get('motion_scope') != 'supervised_primitive_task':
        raise RuntimeError('Primitive route requires the primitive bridge profile')
    if args.route == 'vla' and args.execute and health.get('motion_scope') not in ('supervised_policy_task', 'task_workspace'):
        raise RuntimeError('Managed VLA tasks require a task bridge, not a diagnostic/primitive profile')
    memory = EpisodeMemory(args.memory)
    executive = None
    try:
        executive = TaskExecutive(instruction(args.task), args.route,
            planner or VisualTaskPlanner(args.vlm_port, args.vlm_model), memory,
            scope_key(health, args.route, args.vlm_model), journal,
            mode='live' if args.execute else 'preview', max_replans=args.max_replans,
            max_calls=args.max_planner_calls, snapshot=Path(journal.folder)/'task-state.json')
        log = ExecutiveJournal(journal, executive)
        if args.route == 'primitives':
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
    p.add_argument('--route', choices=['primitives', 'vla'], default='primitives')
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
