"""Summarize measured task, API and mechanical costs from a saved run.

python -m cix_harness.motion_report RUN [--audit ROBOT_AUDIT.jsonl] [--output NEW.json]
"""
import argparse
import json
import math
from pathlib import Path

from .planning import stage_phase


def read_events(path):
    with path.open(encoding='utf-8') as stream:
        for number, line in enumerate(stream, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except ValueError as error:
                    raise ValueError(f'{path.name} line {number}: invalid JSON') from error


def angle_degrees(a, b):
    if not a or not b:
        return None
    norm = math.sqrt(sum(x*x for x in a) * sum(x*x for x in b))
    if not norm:
        return None
    cosine = abs(sum(x*y for x, y in zip(a, b))) / norm
    return math.degrees(2 * math.acos(min(1., cosine)))


def stage_rows(receipts, targets, state, skill, segments):
    rows = []
    previous_eef = state.get('eef_m')
    last_orientation = state.get('eef_xyzw')
    for ordinal, receipt in enumerate(receipts):
        index = receipt.get('stage_index', ordinal)
        stage = receipt.get('stage', {})
        phase = receipt.get('phase')
        if not phase and 0 <= index < len(targets):
            phase = stage_phase(targets, index, skill)
        detail = receipt.get('motion_detail') or segments.get(index, {})
        end = receipt.get('eef_m')
        row = {'index': index, 'kind': stage.get('kind'), 'phase': phase,
               'status': receipt.get('status', 'completed'), 'duration_s': receipt.get('duration_s'),
               'timings_s': receipt.get('timings_s'), 'gripping': receipt.get('gripping'),
               'measured_eef_displacement_m': math.dist(previous_eef, end) if previous_eef and end else None}
        if receipt.get('moveit'):
            row['moveit'] = receipt['moveit']
        if stage.get('kind') == 'move':
            row['requested_rotation_deg'] = angle_degrees(last_orientation, stage.get('xyzw'))
            last_orientation = stage.get('xyzw')
            for key in ('continuous', 'scheduled_seconds', 'legacy_schedule_seconds', 'elapsed_s',
                        'build_seconds', 'loop_timings_s', 'feedback_slowdown_s', 'loop_delay_s',
                        'trajectory_seconds', 'retimed_seconds', 'joint', 'joint_errors_rad',
                        'settling_s', 'bottleneck_joint', 'joint_travel_rad', 'joint_speed_limit_rad_s',
                        'peak_tracking_rad', 'command_samples', 'waypoints', 'planned_eef_path_m'):
                row[key] = detail.get(key)
        if stage.get('kind') == 'gripper':
            row['open'] = stage.get('open')
        rows.append(row)
        previous_eef = end
    return rows


def summarize(folder, audit_path=None):
    folder = Path(folder)
    summary_file = folder/'timing-summary.json'
    summary = json.loads(summary_file.read_text(encoding='utf-8')) if summary_file.exists() else {}
    report = {'run': folder.name, 'wall_seconds': summary.get('wall_seconds'),
              'status': None, 'api_calls': [], 'primitives': [], 'failures': [],
              'timing_summary': {k:v for k,v in summary.get('components', {}).items()
                                 if k.startswith('api.') or k.endswith('.execute')},
              'note': 'Null means unrecorded. API, RPC, server, stage and loop times nest. Endpoint displacement excludes intermediate travel. Requested rotation is a target change.'}
    plans = {}
    observations = {}
    step = {}
    for event in read_events(folder/'events.jsonl'):
        name = event['event']
        plan_id = event.get('plan_id')
        if name == 'observation':
            observations[event['obs_id']] = event['state']
        elif name == 'task_step_dispatch':
            step = event['step']
        elif name == 'planner_api':
            report['api_calls'].append({k:event.get(k) for k in
                ('t','phase','model','response_model','images','status','seconds','total_seconds','auth_seconds',
                 'first_sse_event_seconds','first_output_text_delta_seconds','error_stage','error_type')})
        elif name == 'primitive_plan':
            plan = {'plan_id': plan_id, 'skill': event.get('skill'), 'arm': event.get('selected_arm', step.get('arm')),
                    'step_id': step.get('id'), 'backend': step.get('backend','primitives'),
                    'prepare_status': event.get('status'), 'prepare_timings_s': event.get('timings_s'),
                    'selected_approach_path': event.get('selected_approach_path'),
                    'selected_yaw_rad': event.get('selected_yaw_rad'),
                    'nominal_motion_seconds': event.get('nominal_motion_seconds'),
                    'motion_schedule': event.get('motion_schedule'), 'moveit_full_path': event.get('moveit_full_path'),
                    'execution_status': 'unknown', 'execute_timings_s': None, 'stages': []}
            if event.get('status') == 'ready':
                plans[plan_id] = {'row': plan, 'state': observations.get(event.get('obs_id'), {}),
                                  'targets': event.get('stages', []), 'receipts': [], 'segments': {},
                                  'audit_stages': [], 'has_receipt': False}
            elif event.get('applied') is False:
                plan['execution_status'] = 'not_dispatched'
            if event.get('failure_code'):
                plan['failure_code'] = event['failure_code']
            report['primitives'].append(plan)
        elif name == 'primitive_result':
            if plan_id not in plans:
                # Some old clients only log receipts.
                row = {'plan_id': plan_id, 'skill': event.get('skill'), 'arm': event.get('selected_arm')}
                plans[plan_id] = {'row': row, 'state': {}, 'targets': [], 'segments': {}, 'audit_stages': []}
                report['primitives'].append(row)
            plan = plans[plan_id]
            plan['has_receipt'] = True
            plan['receipts'] = list(event.get('completed_stages', []))
            if event.get('failed_stage'):
                plan['receipts'].append(event['failed_stage'])
            plan['row'].update(execution_status=event.get('status'), execute_timings_s=event.get('timings_s'),
                               failure_code=event.get('failure_code'))
        elif name in ('stop', 'task_step_failed') or (name == 'component_timing' and event.get('succeeded') is False):
            report['failures'].append({k:event[k] for k in
                ('t','event','component','method','duration_s','failure_code','reason') if k in event})
        if name in ('task_finished','finish'):
            report['status'] = event.get('status')
    # The machine audit can span many tasks. Only consume tokens from this run.
    if audit_path:
        active = set()
        pending_legacy_segment = None
        for event in read_events(Path(audit_path)):
            context = event.get('context', {})
            token = event.get('plan_id') or context.get('plan_id')
            name = event['event']
            if name == 'primitive_dispatch':
                active.add(token)
                pending_legacy_segment = None
            if (not token and name in ('segment_arrived', 'segment_arrival_timeout',
                                       'segment_tracking_rejected', 'segment_following_stalled')
                    and len(active) == 1):
                # Older per-arm logs emit an untagged segment immediately before
                # the tagged stage receipt. Require both a dispatch and that receipt.
                pending_legacy_segment = (next(iter(active)), event)
            if name in ('primitive_failed', 'primitive_completed', 'primitive_recovery_available',
                        'primitive_execute_failed'):
                active.discard(token)
            if not token or token not in plans:
                continue
            plan = plans[token]
            if name in ('segment_arrived', 'segment_execution_failed') and 'stage_index' in context:
                plan['segments'][context['stage_index']] = event
            if name in ('primitive_stage_applied', 'primitive_failed'):
                receipt = event if name == 'primitive_stage_applied' else event.get('failed_stage')
                if receipt:
                    if (pending_legacy_segment and pending_legacy_segment[0] == token
                            and receipt['stage']['kind'] == 'move'):
                        index = receipt.get('stage_index', len(plan['audit_stages']))
                        plan['segments'][index] = pending_legacy_segment[1]
                    plan['audit_stages'].append(receipt)
                pending_legacy_segment = None
            if name == 'primitive_failed':
                plan['row']['audit_failure'] = event.get('error')
                if not plan['has_receipt']:
                    plan['row'].update(execution_status='failed_in_robot_audit', execute_timings_s=event.get('timings_s'))
            elif name == 'primitive_completed' and not plan['has_receipt']:
                plan['row'].update(execution_status='completed_in_robot_audit', execute_timings_s=event.get('timings_s'))
                if isinstance(event.get('completed_stages'), list):
                    plan['audit_stages'] = event['completed_stages']
    for plan in plans.values():
        row = plan['row']
        receipts = plan['receipts'] if plan['has_receipt'] else plan['audit_stages']
        row['stages'] = stage_rows(receipts, plan['targets'], plan['state'], row['skill'], plan['segments'])
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run', type=Path)
    parser.add_argument('--audit', type=Path, help='Optional robot-side JSONL. Join by this run’s plan IDs.')
    parser.add_argument('--output', type=Path, help='New JSON file. Defaults to stdout.')
    args = parser.parse_args(argv)
    result = json.dumps(summarize(args.run, args.audit), ensure_ascii=False, indent=2, allow_nan=False)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(result+'\n')
    else:
        print(result)


if __name__ == '__main__':
    main()
