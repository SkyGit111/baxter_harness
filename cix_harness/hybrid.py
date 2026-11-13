"""Per-step backend dispatch with capability gates and serialized physical execution.

The current single-route bridges expose one motion backend. A composite bridge must
explicitly advertise the serialized hybrid protocol; changing a CLI flag never
silently enables additional robot RPCs.
"""
from .contract import ACTION_SPEC
from .primitives import SKILLS
from .task_plan import VLA_SKILLS
from .rpc import HttpRpcClient
from .rollout import predict, rollout, validate_motion_model


def backend_skills(health, policy_health=None, code_tool=None):
    """Return tools supported by the actual bridge and a commissioned policy."""
    scope = health.get('motion_scope')
    if scope == 'supervised_primitive_task':
        supported = {'primitives'}
    elif scope in ('supervised_policy_task', 'task_workspace'):
        supported = {'vla'}
    elif (scope == 'supervised_hybrid_task' and
          health.get('hybrid_protocol') == 'serialized_backends_v1'):
        supported = set(health.get('backend_capabilities', [])) & {'primitives', 'vla'}
    else:
        supported = set()
    result = {}
    if 'primitives' in supported: result['primitives'] = list(SKILLS)
    if 'vla' in supported and policy_health is not None:
        if health.get('action_spec') == policy_health.get('action_spec') == ACTION_SPEC:
            try:
                validate_motion_model(policy_health, health)
            except (ValueError, RuntimeError):
                pass  # An unavailable embodiment adapter is not an executable tool.
            else:
                result['vla'] = list(VLA_SKILLS)
    if code_tool is not None: result['code'] = ['compute']
    return result


def _policy_candidate(health):
    return (health.get('motion_scope') in ('supervised_policy_task', 'task_workspace') or
            (health.get('motion_scope') == 'supervised_hybrid_task' and
             health.get('hybrid_protocol') == 'serialized_backends_v1' and
             'vla' in health.get('backend_capabilities', [])))


def run(args, log, executive, robot=None, model=None, code_tool=None):
    robot = robot or HttpRpcClient(args.robot_port)
    initial = robot.call('healthz')
    if args.execute and initial.get('armed') is not True: raise RuntimeError('Robot is read-only')
    policy_health = None
    if _policy_candidate(initial):
        model = model or HttpRpcClient(args.vla_port)
        try:
            policy_health = model.call('healthz', timeout_s=8)
        except (OSError, RuntimeError, ValueError) as error:
            log('backend_unavailable', backend='vla', reason=type(error).__name__)
    capabilities = backend_skills(initial, policy_health, code_tool)
    if not capabilities: raise RuntimeError('No executable backend is available')
    log('start', route='hybrid', goal=args.task, mode='live' if args.execute else 'preview',
        backend_skills=capabilities)
    scope = initial.get('motion_scope'); fingerprint = initial.get('environment_fingerprint')
    recoveries = 0; empty_grasps = 0; turns = 0; used_views = set(); retract_pending = False
    max_recoveries = getattr(args, 'max_recoveries', 3)
    max_grasp_retries = getattr(args, 'max_grasp_retries', 1)

    def finish(status, evidence=''):
        log('finish', status=status, evidence=evidence)
        return 0 if status in ('done', 'preview') else 2

    def observe(health):
        available = backend_skills(health, policy_health, code_tool)
        observation = robot.call('primitive.observe' if 'primitives' in available else 'env.observe')
        if 'primitives' in available:
            advertised = observation.get('primitive_skills', SKILLS)
            available['primitives'] = [s for s in available['primitives'] if s in advertised]
            if not observation.get('recovery', {}).get('view_indices'):
                available['primitives'] = [s for s in available['primitives'] if s != 'clear_view']
        observation['backend_skills'] = available
        log.camera(observation, 'hybrid')
        return observation, available

    if policy_health and policy_health.get('warmed') is False and 'vla' in capabilities:
        warm_obs = robot.call('env.observe')
        predict(model, warm_obs, 'observe the scene')
        log('warmup', backend='vla', discarded_prediction=True)

    for _ in range(args.max_turns + max_recoveries + 1):
        health = robot.call('healthz')
        if health.get('motion_scope') != scope or health.get('environment_fingerprint') != fingerprint:
            raise RuntimeError('Robot profile changed during the task; reinitialize the task')
        if args.execute and health.get('armed') is not True: raise RuntimeError('Robot became read-only')
        obs, available = observe(health)
        views = [i for i in obs.get('recovery', {}).get('view_indices', []) if i not in used_views]
        view_index = None; recovering = retract_pending
        if retract_pending:
            if obs.get('state', {}).get('gripping') is not False:
                return finish('grasp_state_changed_during_recovery')
            decision = {'decision': 'act', 'backend': 'primitives',
                        'skill': 'clear_view' if views else 'retreat', 'uv': None,
                        'evidence': 'Retract after confirmed empty grasp before fresh localization'}
        else:
            decision = executive.choose(obs)
        log('decision', **decision)
        if decision['decision'] == 'blocked' and decision.get('reason') == 'occluded':
            log('recoverable_block', **decision)
            if not executive.recovery_allowed('occluded'): return finish('recovery_policy_stop')
            if 'clear_view' not in available.get('primitives', []):
                return finish('occlusion_recovery_unavailable', decision['evidence'])
            decision = {'decision': 'act', 'backend': 'primitives', 'skill': 'clear_view', 'uv': None,
                        'evidence': 'Move aside and reobserve the occluded target'}
        if decision['decision'] != 'act':
            return finish('done' if decision['decision'] == 'done' else 'blocked', decision.get('evidence', ''))
        backend = decision.get('backend'); skill = decision.get('skill')
        if skill not in available.get(backend, []):
            raise RuntimeError('Planner selected a backend/skill not advertised by this bridge')
        if turns >= args.max_turns: return finish('turn_budget_exhausted')

        if backend == 'code':
            context = {'goal': args.task, 'state': obs.get('state', {}), 'obs_id': obs['id'],
                       'feedback': getattr(executive, 'feedback', None)}
            try:
                result = code_tool.execute(decision.get('code', ''), context)
            except ValueError as error:
                result = {'status': 'failed', 'failure_code': 'code_error', 'message': str(error)[:300], 'applied': False}
            if not isinstance(result, dict) or result.get('applied') is not False:
                raise RuntimeError('Code helper must return data without applying physical actions')
            log('code_result', **result); turns += 1
            if not args.execute: return finish('preview')
            if result.get('status') == 'failed':
                if recoveries >= max_recoveries: return finish('recovery_budget_exhausted')
                recoveries += 1; log('recovery', kind='code_replan', attempt=recoveries)
            elif result.get('status') != 'completed':
                raise RuntimeError('Unknown code helper outcome')
            continue

        if backend == 'vla':
            policy_health = model.call('healthz', timeout_s=8)
            validate_motion_model(policy_health, health)
            if policy_health.get('action_spec')!=ACTION_SPEC:raise RuntimeError('VLA action contract changed during task')
            capacity = health.get('max_action_segment', 0) if args.execution_mode == 'segment' else health.get('max_action_window', 1)
            if args.action_window > min(policy_health.get('max_action_window', 1), capacity):
                raise RuntimeError('Requested VLA action window exceeds backend capacity')
            if not args.execute:
                fresh = robot.call('env.observe'); log.camera(fresh, 'vla')
                action = predict(model, fresh, decision['subtask'])
                log('preview', backend='vla', obs_id=fresh['id'], action=action)
                return finish('preview')
            if health.get('motion_remaining') == 0: return finish('robot_motion_budget_exhausted')
            rollout(robot, model, decision['subtask'], args.steps, log, args.action_window,
                    monitor=executive.monitor, execution_mode=args.execution_mode)
            if executive.monitor.evidence().get('drop_detected'):
                raise RuntimeError('Held object lost during VLA execution')
            turns += 1; used_views.clear()
            continue

        if skill == 'clear_view':
            if not views: return finish('no_unused_observation_pose')
            if not recovering:
                if recoveries >= max_recoveries: return finish('recovery_budget_exhausted')
                recoveries += 1
            view_index = views[0]; used_views.add(view_index)
            log('recovery', kind='clear_view', view_index=view_index, attempt=recoveries)
        kwargs = {'kwargs': {'view_index': view_index}} if view_index is not None else {}
        prepared = robot.call('primitive.prepare', (obs['id'], skill, decision.get('uv')), timeout_s=60, **kwargs)
        log('primitive_plan', **prepared)
        if prepared.get('status') == 'rejected':
            if prepared.get('recoverable') is not True or prepared.get('applied') is not False:
                return finish(prepared.get('failure_code', 'preflight_rejected'))
            if recoveries >= max_recoveries: return finish('recovery_budget_exhausted')
            recoveries += 1; log('recovery', kind='reobserve', cause=prepared['failure_code'], attempt=recoveries)
            continue
        if not args.execute: return finish('preview')
        # A lost response may mean physical motion happened. Never retry this call.
        receipt = robot.call('primitive.execute', (prepared['plan_id'],), timeout_s=600)
        if receipt.get('applied') is not True or receipt.get('plan_id') != prepared['plan_id']:
            raise RuntimeError('Invalid primitive acknowledgement')
        log('primitive_result', **receipt); turns += 1
        if receipt.get('status') == 'failed':
            if (skill != 'pick' or receipt.get('failure_code') != 'empty_grasp' or
                receipt.get('recoverable') is not True or receipt.get('stopped') is not True or
                receipt.get('gripping') is not False):
                raise RuntimeError('Primitive failure is not safe to recover automatically')
            if empty_grasps >= max_grasp_retries or recoveries >= max_recoveries:
                return finish('grasp_recovery_budget_exhausted')
            if not executive.recovery_allowed('empty_grasp'): return finish('recovery_policy_stop')
            empty_grasps += 1; recoveries += 1; retract_pending = True; used_views.clear()
            log('recovery', kind='empty_grasp_retraction', attempt=empty_grasps)
            continue
        if receipt.get('status', 'completed') != 'completed': raise RuntimeError('Unknown primitive outcome')
        if recovering: retract_pending = False
        elif skill != 'clear_view': used_views.clear()
    return finish('planning_budget_exhausted')
