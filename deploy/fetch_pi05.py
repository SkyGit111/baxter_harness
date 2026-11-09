"""Download a pinned official LeRobot checkpoint; validate all files before use."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time

parser = argparse.ArgumentParser()
parser.add_argument('--manifest', required=True)
parser.add_argument('--dir', required=True)
parser.add_argument('--endpoint', default='https://huggingface.co')
parser.add_argument('--metadata-only', action='store_true')
args = parser.parse_args()
manifest = json.loads(Path(args.manifest).read_text())
root = Path(args.dir)
root.mkdir(parents=True, exist_ok=True)

def valid(path, item):
    if not path.is_file() or path.stat().st_size != item['size']:
        return False
    if 'lfs' in item:
        digest = hashlib.sha256()
        expected = item['lfs']['sha256']
    else:
        digest = hashlib.sha1(('blob %d\0' % item['size']).encode())
        expected = item['blobId']
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest() == expected

for item in manifest['siblings']:
    if args.metadata_only and item['size'] > 100_000_000:
        continue
    name = item['rfilename']
    if name == '.gitattributes':
        continue
    if Path(name).is_absolute() or '..' in Path(name).parts:
        raise ValueError('Invalid checkpoint path')
    target = root / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if valid(target, item):
        print('Verified existing', name, flush=True)
        continue
    part = target.with_name(target.name + '.part')
    url = args.endpoint.rstrip('/') + '/lerobot/pi05-libero/resolve/' + manifest['sha'] + '/' + name
    print('Downloading', name, item['size'], flush=True)
    for attempt in range(5):
        # A fresh invocation recomputes the resume offset after a partial CDN
        # transfer, and a fresh URL avoids reusing an expired signed redirect.
        result = subprocess.run(['curl', '-fLsS', '--retry', '3', '--connect-timeout', '15',
                                 '-C', '-', '-o', str(part), url + '?download=true&request_time=' + str(int(time.time()))])
        if result.returncode == 0:
            break
        if attempt == 4:
            result.check_returncode()
        print('Resuming after curl error', result.returncode, flush=True)
        time.sleep(min(2**attempt, 16))
    if not valid(part, item):
        raise RuntimeError('Checkpoint integrity failed: ' + name)
    part.replace(target)
    print('Verified', name, flush=True)
if not args.metadata_only:
    (root / 'verified-revision.txt').write_text(manifest['sha'] + '\n')
print('Metadata complete' if args.metadata_only else 'Checkpoint complete', flush=True)
