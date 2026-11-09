"""Offline coordinate audit. Reads recorded data only; never connects to hardware."""
import argparse
import json
import math
from pathlib import Path
import struct

from .pi05_contract import physical_delta, rotate, state8


def state_statistics(path):
    """Read the small checkpoint normalizer artifact without loading Torch/weights."""
    data = Path(path).read_bytes()
    size = struct.unpack('<Q', data[:8])[0]
    header = json.loads(data[8:8+size])
    result = {}
    for name in ('mean', 'std', 'q01', 'q99', 'min', 'max'):
        entry = header['observation.state.'+name]
        if entry['dtype'] != 'F32' or entry['shape'] != [8]:
            raise ValueError('Expected eight float32 state statistics')
        start, end = entry['data_offsets']
        result[name] = list(struct.unpack('<8f', data[8+size+start:8+size+end]))
    return result


def state_report(values, stats):
    normalized = [(x-m)/(s+1e-8) for x, m, s in zip(values, stats['mean'], stats['std'])]
    names = ['x', 'y', 'z', 'rotvec_x', 'rotvec_y', 'rotvec_z', 'finger_left', 'finger_right']
    return {
        'state8': values, 'normalized_state8': normalized,
        # Exactly the pinned tokenizer's bin boundaries, including -1 below range.
        'state_tokens8': [min(255, max(-1, math.floor((v+1)*128))) for v in normalized],
        'outside_training_q01_q99': [name for name, v, lo, hi in
                                    zip(names, values, stats['q01'], stats['q99']) if not lo <= v <= hi],
        'outside_training_min_max': [name for name, v, lo, hi in
                                    zip(names, values, stats['min'], stats['max']) if not lo <= v <= hi],
        'note': 'Distribution indicators, not hard workspace limits or proof of the correct frame.',
    }


def motion_report(receipt):
    before = receipt['endpoint_before_m']
    commanded = [v-b for v, b in zip(receipt['endpoint_target_m'], before)]
    measured = [v-b for v, b in zip(receipt['endpoint_after_m'], before)]
    norm = lambda a: math.sqrt(sum(x*x for x in a))
    product = norm(commanded)*norm(measured)
    cosine = sum(a*b for a, b in zip(commanded, measured))/product if product > 1e-12 else None
    return {'obs_id': receipt['obs_id'], 'commanded_delta_m': commanded,
            'measured_delta_m': measured,
            'direction_error_deg': math.degrees(math.acos(max(-1, min(1, cosine)))) if cosine is not None else None,
            'position_error_m': norm([a-b for a, b in zip(commanded, measured)])}


def audit(run_dir, traces_dir, normalizer):
    events = [json.loads(line) for line in (Path(run_dir)/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    observations = {e['obs_id']: e for e in events if e['event'] == 'observation'}
    predictions = {e['obs_id']: e for e in events if e['event'] == 'prediction'}
    stats = state_statistics(normalizer)
    rows = []
    matched = set()
    for path in sorted(Path(traces_dir).glob('*.json')):
        trace = json.loads(path.read_text(encoding='utf-8'))
        oid = trace.get('obs_id')
        if oid not in predictions or 'raw_libero_chunk' not in trace:
            continue
        matched.add(oid)
        converted = [physical_delta(a) for a in trace['raw_libero_chunk']]
        actual = predictions[oid]['actions']
        if len(converted) < len(actual):
            raise ValueError('Recorded policy chunk shorter than dispatched predictions')
        max_error = max(abs(x-y) for a, b in zip(converted, actual) for x, y in zip(a, b))
        profile = trace['profile']
        recomputed = state8(observations[oid]['state'], profile)
        q = profile['policy_from_base_xyzw']
        row = {'obs_id': oid, 'trace_file': path.name,
               'action_conversion_max_error': max_error,
               'state_reconstruction_max_error': max(abs(x-y) for x, y in zip(recomputed, trace['state8'])),
               'raw_first_xyz': trace['raw_libero_chunk'][0][:3],
               'physical_first_policy_xyz_m': converted[0][:3],
               'physical_first_base_xyz_m': rotate([-q[0], -q[1], -q[2], q[3]], converted[0][:3]),
               **state_report(trace['state8'], stats)}
        rows.append(row)
    return {'run': str(run_dir), 'prediction_count': len(predictions),
            'missing_trace_obs_ids': sorted(set(predictions)-matched), 'predictions': rows,
            'motions': [motion_report(e) for e in events if e['event'] == 'applied' and 'endpoint_after_m' in e],
            'checkpoint_state_statistics': stats,
            'limitation': 'Verifies recorded signs and numeric transforms, not camera calibration or target correctness.'}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', required=True)
    p.add_argument('--traces', required=True)
    p.add_argument('--normalizer', required=True)
    p.add_argument('--output', required=True)
    args = p.parse_args()
    result = audit(args.run, args.traces, args.normalizer)
    Path(args.output).write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({'predictions': len(result['predictions']), 'motions': result['motions'],
                      'missing_trace_obs_ids': result['missing_trace_obs_ids']}, indent=2))


if __name__ == '__main__':
    main()
