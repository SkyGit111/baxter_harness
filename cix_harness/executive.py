"""Shared task executive: semantic plans, measured checkpoints, recovery and memory."""
import copy
import math

from .task_plan import EXPECTED, VLA_SKILLS, compact_state, validate_plan, validate_check
from .primitives import SKILLS
from .task_monitor import TaskMonitor


class TaskExecutive:
    def __init__(self, goal, route, planner, memory, scope, log, mode='preview',
                 max_replans=2, max_calls=32, snapshot=None):
        self.goal, self.route, self.planner = goal, route, planner
        self.memory, self.log, self.mode, self.snapshot = memory, log, mode, snapshot
        self.episode = memory.start(scope, route, goal, mode)
        self.memories = memory.retrieve(goal, scope)
        self.max_replans, self.max_calls = max_replans, max_calls
        self.calls = 0; self.replans = 0; self.revision = 0
        self.steps = []; self.index = 0; self.verified = []; self.archived = []
        self.failures = []; self.recoveries = []; self.pending_failure = None; self.feedback = None
        self.dispatch = None; self.primitive_ok = False; self.last_action_obs = None
        self.predictions = {}; self.monitor = TaskMonitor(); self.released = False
        self.applied = 0; self.origin = None; self.attempts = 0; self.step_failures = 0
        self.task_verified = False; self.closed = False; self.last_obs = None; self.status = 'running'
        self.last_target = None
        self.emit('task_started', episode=self.episode, goal=goal, route=route, mode=mode,
                  memory_matches=self.memories, cross_run_memory_enabled=scope is not None)

    @property
    def current(self):
        return self.steps[self.index] if self.index < len(self.steps) else None

    def emit(self, event, **fields):
        self.log(event, **fields)
        if self.snapshot is not None:
            import json
            state = {'episode': self.episode, 'goal': self.goal, 'route': self.route, 'mode': self.mode,
                     'revision': self.revision, 'steps': self.steps, 'verified': self.verified,
                     'archived': self.archived, 'feedback': self.feedback, 'failures': self.failures,
                     'recovery_attempts': self.recoveries, 'last_target': self.last_target,
                     'planner_calls': self.calls, 'replans': self.replans,
                     'task_verified': self.task_verified, 'closed': self.closed, 'status': self.status}
            pending = self.snapshot.with_name(self.snapshot.name+'.pending')
            pending.write_text(json.dumps(state, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
            pending.replace(self.snapshot)

    def reset_step(self, obs):
        self.dispatch = None; self.primitive_ok = False; self.last_action_obs = None
        self.monitor = TaskMonitor(); self.released = False; self.predictions = {}
        self.applied = 0; self.attempts = 0; self.step_failures = 0
        self.last_target = None
        self.origin = obs.get('state', {}).get('eef_m')
        self.monitor.observe(obs.get('state', {}))

    def skills(self, obs):
        if self.route == 'vla': return list(VLA_SKILLS)
        return [s for s in obs.get('primitive_skills', SKILLS)
                if s in SKILLS and (s != 'clear_view' or obs.get('recovery', {}).get('view_indices'))]

    def call_planner(self, method, obs, context):
        if self.calls >= self.max_calls: raise RuntimeError('Task planner call budget exhausted')
        self.calls += 1
        reply = getattr(self.planner, method)(obs, context)
        self.emit('task_model_reply', phase=method, obs_id=obs['id'], reply=reply)
        return reply

    def build_plan(self, obs, cause):
        if self.steps:
            if self.replans >= self.max_replans: return False
            self.replans += 1
        skills = self.skills(obs)
        context = {'goal': self.goal, 'route': self.route, 'state': compact_state(obs),
                   'available_skills': skills, 'skill_expect': {s: EXPECTED[s] for s in skills},
                   'verified': [{k: s[k] for k in ('skill', 'target')} for s in self.verified[-4:]],
                   'cause': cause, 'memory': self.memories[:1]}
        # One read-only format/contract repair. Never turn an invalid plan into a motor command.
        for repair in range(2):
            try:
                reply = self.call_planner('plan', obs, context)
                steps = validate_plan(reply, skills, obs.get('state', {}).get('gripping'))
                break
            except ValueError as error:
                self.emit('task_plan_rejected', message=str(error))
                if repair: raise
                context['format_error'] = str(error)
        if self.steps: self.archived.append(copy.deepcopy(self.steps))
        self.revision += 1
        self.steps = [{**s, 'id': 'r{}s{}'.format(self.revision, i+1), 'status': 'pending'}
                      for i, s in enumerate(steps)]
        self.index = 0; self.pending_failure = None; self.feedback = None
        self.reset_step(obs)
        self.emit('task_plan', revision=self.revision, steps=self.steps, cause=cause)
        return True

    def observe(self, obs):
        self.last_obs = obs
        if self.current: self.monitor.observe(obs.get('state', {}))

    def step_evidence(self, obs):
        result = {**self.monitor.evidence(), 'primitive_completed': self.primitive_ok,
                  'commands_applied': self.applied, 'confirmed_release': self.released,
                  'fresh_after_action': self.last_action_obs is not None and obs['id'] != self.last_action_obs}
        position = obs.get('state', {}).get('eef_m')
        result['distance_from_start_m'] = round(math.dist(position, self.origin), 3) if position and self.origin else None
        return result

    def can_verify(self, obs):
        if self.mode != 'live' or not self.current: return False
        state = obs.get('state', {}); skill = self.current['skill']; evidence = self.step_evidence(obs)
        if not evidence['fresh_after_action']: return False
        if self.route == 'primitives':
            if not self.primitive_ok: return False
            if skill == 'pick': return state.get('gripping') is True
            if skill in ('place', 'stack', 'open'):
                return state.get('gripping') is False and state.get('gripper_percent', 0) > 70
            return not evidence['drop_detected']
        if not self.applied or evidence['drop_detected']: return False
        if skill == 'pick': return evidence['held_and_lifted'] and state.get('gripping') is True
        if skill in ('place', 'stack'): return self.released and state.get('gripping') is False
        if skill == 'lift':
            position = state.get('eef_m')
            return bool(position and self.origin and position[2]-self.origin[2] >= .025)
        return (evidence['distance_from_start_m'] or 0) >= .02

    def terminal(self, kind, evidence, reason='none'):
        if self.route == 'primitives':
            # The existing primitive executor understands these terminal categories.
            if reason == 'no_progress': reason = 'unreachable'
            return {'decision': kind, 'reason': reason, 'skill': '', 'uv': None, 'evidence': evidence}
        return {'decision': kind, 'subtask': '', 'evidence': evidence}

    def final_state_consistent(self, obs):
        state = obs.get('state', {})
        # Recheck the latest grasp/release after any final view-clearing motion.
        for step in reversed(self.verified):
            if step['skill'] == 'pick': return state.get('gripping') is True
            if step['skill'] in ('place', 'stack', 'open'):
                return state.get('gripping') is False and state.get('gripper_percent', 0) > 70
            if step['skill'] == 'close': return state.get('gripper_percent', 100) < 70
        return True

    def choose(self, obs):
        self.observe(obs)
        if not self.steps and not self.build_plan(obs, 'new_task'):
            return self.terminal('blocked', 'No valid plan', 'unreachable')
        if self.pending_failure:
            cause = self.pending_failure; self.pending_failure = None
            policy = self.current['on_failure'] if self.current else 'stop'
            if policy == 'stop': return self.terminal('blocked', 'Step recovery policy requests stop', 'unreachable')
            if policy == 'replan' or self.step_failures >= 2:
                if not self.build_plan(obs, cause):
                    return self.terminal('blocked', 'Replanning budget exhausted', 'no_progress')
        rejected = 0
        for _ in range(16):
            if self.current and self.route == 'vla' and (self.attempts >= 3 or obs.get('harness_feedback', {}).get('stalled')):
                self.failures.append('no_progress')
                if self.current['on_failure'] == 'stop':
                    return self.terminal('blocked', 'Step recovery policy requests stop after no verified progress', 'no_progress')
                if not self.build_plan(obs, 'no_progress'):
                    return self.terminal('blocked', 'No verified progress after bounded replanning', 'no_progress')
                # Consume this observation's stall notice once, not once per internal checkpoint.
                obs = {**obs, 'harness_feedback': {}}
            context = {'goal': self.goal, 'route': self.route, 'state': compact_state(obs),
                       'current': self.current,
                       'remaining': [{'skill': s['skill'], 'target': s['target']} for s in self.steps[self.index+1:]],
                       'verified': [{'skill': s['skill'], 'target': s['target']} for s in self.verified[-4:]],
                       'step_evidence': self.step_evidence(obs), 'feedback': self.feedback}
            try:
                answer = validate_check(self.call_planner('check', obs, context), self.route, self.current)
            except ValueError as error:
                if rejected: raise
                rejected += 1; self.feedback = {'failure_code': 'invalid_checkpoint', 'message': str(error)}
                continue
            kind = answer['decision']
            if kind == 'replan':
                if not self.build_plan(obs, answer['reason']+': '+answer['evidence']):
                    return self.terminal('blocked', 'Replanning budget exhausted', 'no_progress')
                continue
            if kind == 'blocked':
                if answer['reason'] == 'occluded' and self.current and self.current['on_failure'] == 'replan':
                    self.failures.append('occluded')
                    if self.build_plan(obs, 'occluded: '+answer['evidence']): continue
                    return self.terminal('blocked', 'Occlusion replanning budget exhausted', 'no_progress')
                return self.terminal('blocked', answer['evidence'], answer['reason'])
            if kind == 'task_done':
                if self.current or not self.verified or self.mode != 'live' or not self.final_state_consistent(obs):
                    self.feedback = {'failure_code': 'unverified_task_completion'}
                    self.emit('task_checkpoint_rejected', **self.feedback)
                    if rejected: return self.terminal('blocked', 'Task completion has no verified support', 'no_progress')
                    rejected += 1; continue
                self.task_verified = True
                self.emit('task_verified', evidence=answer['evidence'], obs_id=obs['id'])
                return self.terminal('done', answer['evidence'])
            if kind == 'step_done':
                if not self.can_verify(obs):
                    self.feedback = {'failure_code': 'missing_measured_postcondition', 'expect': self.current['expect']}
                    self.emit('task_checkpoint_rejected', **self.feedback)
                    if rejected: return self.terminal('blocked', 'Step outcome is not supported by telemetry', 'no_progress')
                    rejected += 1; continue
                self.current.update(status='verified', evidence=answer['evidence'], obs_id=obs['id'])
                self.verified.append(copy.deepcopy(self.current));self.index += 1
                self.feedback = None; self.reset_step(obs)
                self.emit('task_step_verified', step=self.verified[-1])
                continue
            if self.primitive_ok:
                # A complete primitive is not automatically repeated if the VLM dislikes the result.
                if not self.build_plan(obs, 'primitive_completed_but_outcome_not_accepted'):
                    return self.terminal('blocked', 'Completed primitive needs a revised plan', 'no_progress')
                continue
            self.current['status'] = 'active';self.attempts += 1
            self.dispatch = self.current['id'];self.last_action_obs = obs['id']
            self.last_target = {'obs_id': obs['id'], 'uv': answer['uv'], 'target': self.current['target']}
            self.emit('task_step_dispatch', step=self.current, obs_id=obs['id'], evidence=answer['evidence'])
            if self.route == 'primitives':
                return {'decision': 'act', 'reason': 'none', 'skill': self.current['skill'],
                        'uv': answer['uv'], 'evidence': answer['evidence']}
            return {'decision': 'act', 'subtask': self.current['instruction'], 'evidence': answer['evidence']}
        return self.terminal('blocked', 'Internal planning budget exhausted', 'no_progress')

    def event(self, name, fields):
        """Consume executor facts, not model claims. Automatic recovery is not task completion."""
        if name == 'finish':
            if fields.get('status') != 'done': self.failures.append(fields.get('status', 'stopped'))
            self.finish('verified' if fields.get('status') == 'done' and self.task_verified else 'blocked')
            return
        if name == 'recovery':
            self.recoveries.append(fields.get('kind', 'unknown'))
            self.emit('task_recovery', recovery=fields, step_id=self.current['id'] if self.current else None)
            return
        if name == 'recoverable_block':
            self.failures.append(fields['reason'])
            self.feedback = {'failure_code': fields['reason']}
            return
        if not self.current or self.dispatch != self.current['id']: return
        if name == 'primitive_plan':
            if fields.get('status') == 'rejected': self.failed(fields.get('failure_code', 'preflight_rejected'))
        elif name == 'primitive_result':
            if fields.get('skill') != self.current['skill']: return
            if fields.get('status') == 'failed': self.failed(fields.get('failure_code', 'execution_failed'))
            elif fields.get('status') == 'completed' and fields.get('applied') is True:
                self.primitive_ok = True; self.current['status'] = 'awaiting_verification'; self.dispatch = None
                # Verify_grasp stages belong to this primitive. Do not mistake a planned release for a drop.
                if self.current['skill'] in ('place', 'stack'): self.monitor = TaskMonitor()
                self.feedback = {'status': 'completed', 'skill': self.current['skill']}
                self.emit('task_step_executed', step=self.current)
        elif name == 'prediction':
            self.predictions = {fields['obs_id']: fields['actions']}
        elif name == 'applied':
            actions = self.predictions.pop(fields.get('obs_id'), None)
            if actions is None: raise RuntimeError('Execution receipt lacks a matching policy prediction')
            count = fields.get('applied_actions', 1)
            if type(count) is not int or not 1 <= count <= len(actions): raise RuntimeError('Invalid applied action count')
            items = fields.get('receipts')
            pairs = zip(actions[:count], items) if items is not None else [(actions[count-1], fields)]
            for action, receipt in pairs:
                if (self.monitor.last_state.get('gripping') is True and receipt.get('gripping') is False
                    and receipt.get('gripper_percent', 0) > 70 and action[6] >= .5):
                    self.released = True
                self.monitor.applied(receipt, action)
            self.applied += count;self.last_action_obs = fields['obs_id']
            self.current['status'] = 'awaiting_verification'
            self.feedback = {'status': 'executed', 'target_reached': fields.get('target_reached')}
            self.emit('task_step_executed', step=self.current, commands_applied=self.applied)

    def failed(self, code):
        self.failures.append(code);self.pending_failure = code;self.step_failures += 1
        self.feedback = {'failure_code': code, 'previous_target': self.last_target}
        self.primitive_ok = False;self.dispatch = None
        self.current['status'] = 'needs_recovery'
        self.emit('task_step_failed', step=self.current, failure_code=code)

    def recovery_allowed(self, code):
        # Final scene verification can also need a fresh unobstructed view.
        return self.current is None or self.current['on_failure'] != 'stop'

    def finish(self, status):
        if self.closed: return
        if self.mode != 'live': status = 'preview'
        if status == 'verified' and not self.task_verified: status = 'unknown'
        self.memory.finish(self.episode, status, self.verified, self.failures, self.recoveries)
        self.status = status; self.closed = True;self.emit('task_finished', status=status, episode=self.episode)


class ExecutiveJournal:
    """Add executive state updates while retaining the existing raw execution journal."""
    def __init__(self, journal, executive):
        self.journal, self.executive = journal, executive

    def __call__(self, event, **fields):
        self.journal(event, **fields)
        self.executive.event(event, fields)

    def camera(self, obs, role):
        self.journal.camera(obs, role)
        self.executive.observe(obs)
