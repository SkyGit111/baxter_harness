"""Replay archived observations through Pi05; this program has no motor RPC calls.

Run from the project root on the board. Both the robot and policy must already
be read-only. A historical profile is used with commissioned=false, then restored.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from cix_harness.contract import image_data
from cix_harness.rpc import HttpRpcClient


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True)
    parser.add_argument('--recorded-profile', required=True)
    parser.add_argument('--active-profile', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    robot, policy = HttpRpcClient(9102), HttpRpcClient(9101)

    def require_readonly():
        if robot.call('healthz')['armed'] is not False or policy.call('healthz')['motion_ready'] is not False:
            raise RuntimeError('Both endpoints must remain read-only for replay')

    require_readonly()
    run_dir = Path(args.run)
    events = [json.loads(x) for x in (run_dir/'events.jsonl').read_text().splitlines()]
    observations = {e['obs_id']: e for e in events if e['event'] == 'observation'}
    predictions = [e for e in events if e['event'] == 'prediction']
    selected = [predictions[0], predictions[-1]]
    active = Path(args.active_profile)
    original = active.read_bytes()
    if json.loads(original).get('commissioned') is True:
        raise RuntimeError('The on-disk profile must also be read-only')
    profile = json.loads(Path(args.recorded_profile).read_text())
    profile['commissioned'] = False
    results = []
    try:
        active.write_text(json.dumps(profile))
        policy.call('vla.reload_profile')
        for prediction in selected:
            entry = observations[prediction['obs_id']]
            scene = run_dir/entry['file']
            wrist = scene.with_name(scene.stem+'-wrist'+scene.suffix)
            obs = {'id': 'readonly-replay-'+prediction['obs_id'], 'state': entry['state'],
                   'task_descriptions': prediction['subtask'],
                   'image': image_data(scene.read_bytes()), 'wrist_image': image_data(wrist.read_bytes())}
            for seed in [42, 43, 44, 42]:
                require_readonly()
                result = policy.call('vla.predict', (obs, {'action_window': 16, 'diagnostic_seed': seed}))
                if result['motion_ready'] is not False:
                    raise RuntimeError('Diagnostic result unexpectedly commissioned')
                row = {'original_obs_id': prediction['obs_id'], 'seed': seed, **result}
                results.append(row)
                Path(args.output).write_text(json.dumps(results, indent=2))
                print(json.dumps({'source': entry['file'], 'seed': seed,
                                  'first_xyz_m': result['actions'][0][:3], 'seconds': result['seconds']}), flush=True)
    finally:
        active.write_bytes(original)
        policy.call('vla.reload_profile')
        require_readonly()
        print('Replay ended. Original profile restored; both endpoints read-only.', flush=True)


if __name__ == '__main__':
    main()
