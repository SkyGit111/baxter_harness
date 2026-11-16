"""Build deterministic source and separate private site-config ZIPs without networking.

Source is read from Git INDEX blobs, never uncommitted worktree contents. New code
must first be reviewed and staged; this command never stages or commits files.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRS = {'cix_harness', 'tests', 'config', 'deploy', 'docs', 'scripts', 'integrations', 'third_party', '.github'}
SOURCE_EXTENSIONS = {'.py', '.md', '.json', '.sh', '.ps1', '.txt', '.lock', '.toml', '.yaml', '.yml', '.ini', '.cfg'}
FORBIDDEN_PARTS = {'.local', '.git', 'runs', 'models', 'vendor', 'dist', '__pycache__', 'wheels', 'node_modules', '.venv', 'venv'}
SECRET_KEYS = {'access_token', 'refresh_token', 'id_token', 'api_key', 'password', 'passwd', 'private_key',
               'client_secret', 'authorization', 'openai_api_key'}
SYNTHETIC_FIXTURE = 'tests/fixtures/white-pixel.png'
SYNTHETIC_FIXTURE_SHA256 = 'e878950f8091ec010cf5cc723bdea027a8539cf7147cfea199c2f666232dcd4e'


def allowed_source(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:return False
    if any(part.lower() in FORBIDDEN_PARTS for part in path.parts):return False
    base = path.name.lower()
    if base.startswith('.env') or '.local.' in base or base in {'credentials.json', 'chatgpt-session.json', 'auth.json'}:
        return False
    if len(path.parts) == 1:return name in {'README.md', '.gitignore', '.gitattributes', '.dockerignore'}
    if path.parts[0] not in SOURCE_DIRS:return False
    if name == SYNTHETIC_FIXTURE:return True
    if name in ('deploy/Dockerfile.vla', 'deploy/Dockerfile.vlm'):return True
    if path.parts[0] == 'config' and not base.endswith('.example.json'):return False
    return path.suffix.lower() in SOURCE_EXTENSIONS or (path.parts[0] == 'third_party' and 'license' in base)


def validate_source(name, raw):
    if name == SYNTHETIC_FIXTURE:
        if hashlib.sha256(raw).hexdigest() != SYNTHETIC_FIXTURE_SHA256:
            raise ValueError('The only allowed image is the known synthetic white-pixel fixture')
        return raw
    if len(raw) > 2 * 1024 * 1024:raise ValueError('Source file exceeds 2 MiB: ' + name)
    text = raw.decode('utf-8-sig')
    if '\x00' in text or re.search(r'^-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----', text, re.MULTILINE):
        raise ValueError('Binary data or private key found in source selection')
    if re.search(r'\bsk-[A-Za-z0-9_-]{24,}', text):
        raise ValueError('Possible API credential found in source selection')
    if PurePosixPath(name).suffix.lower() == '.json':
        _reject_secrets(json.loads(text))
    return raw


def tracked_sources(root):
    output = subprocess.run(['git', '-C', str(root), 'ls-files', '--stage', '-z'], check=True, capture_output=True).stdout
    result = {}
    for entry in output.decode('utf-8').split('\0'):
        if not entry:continue
        metadata, name = entry.split('\t', 1)
        mode, digest, stage = metadata.split()
        if stage != '0':raise ValueError('Resolve Git index conflicts before bundling')
        if not allowed_source(name):continue
        if mode not in ('100644', '100755'):raise ValueError('Source selection includes a symlink or submodule')
        # Read the exact blob named by the index snapshot, not current disk bytes.
        raw = subprocess.run(['git', '-C', str(root), 'cat-file', 'blob', digest], check=True, capture_output=True).stdout
        result[name] = validate_source(name, raw)
    if not result:raise ValueError('No tracked source files selected')
    return result


def _reject_secrets(value):
    if isinstance(value, dict):
        for key, item in value.items():
            if str(key).lower() in SECRET_KEYS:raise ValueError('Credential field is forbidden in a site-config bundle')
            _reject_secrets(item)
    elif isinstance(value, list):
        for item in value:_reject_secrets(item)
    elif isinstance(value, str):
        if value.startswith(('data:image/', 'sk-', 'Bearer ')) or 'PRIVATE KEY-----' in value:
            raise ValueError('Credentials or image data are forbidden in a site-config bundle')


def site_configs(paths):
    result = {}
    for path in paths:
        path = Path(path)
        if not path.name.endswith('.local.json') or path.is_symlink():
            raise ValueError('Only explicitly named .local.json site configurations may be bundled')
        raw = path.read_bytes()
        if len(raw) > 1024 * 1024:raise ValueError('Site configuration exceeds 1 MiB')
        def finite(_value):raise ValueError('Nonfinite site configuration')
        value = json.loads(raw.decode('utf-8-sig'), parse_constant=finite)
        if not isinstance(value, dict) or not ('arms' in value or 'primitives' in value):
            raise ValueError('Expected a robot site configuration, not an account or arbitrary JSON file')
        _reject_secrets(value)
        name = 'config/' + path.name
        if name in result:raise ValueError('Duplicate site configuration filename')
        result[name] = raw
    return result


def write_bundle(output, files, kind, revision):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {'format': 'baxter-harness-bundle-v1', 'kind': kind, 'source_revision': revision,
                'files': {name: {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
                          for name, raw in sorted(files.items())}}
    # Fixed ordering, timestamps and permissions make equal inputs byte-reproducible.
    contents = dict(files)
    contents['bundle.manifest.json'] = (json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + '\n').encode('utf-8')
    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for name, raw in sorted(contents.items()):
            record = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            record.create_system = 3
            record.external_attr = (0o100600 if kind == 'private-site-config' else 0o100644) << 16
            record.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(record, raw)
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    output.with_name(output.name + '.sha256').write_text(digest + '  ' + output.name + '\n', encoding='ascii')
    return {'file': str(output), 'sha256': digest, 'files': len(files), 'bytes': output.stat().st_size, 'kind': kind}


def verify_archive(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names):raise ValueError('Archive contains duplicate names')
        if sum(item.file_size for item in archive.infolist()) > 64 * 1024 * 1024:
            raise ValueError('Archive exceeds source bundle size limit')
        if any(stat.S_ISLNK(item.external_attr >> 16) for item in archive.infolist()):
            raise ValueError('Archive symlinks are forbidden')
        if archive.getinfo('bundle.manifest.json').file_size > 1024 * 1024:
            raise ValueError('Manifest exceeds size limit')
        manifest = json.loads(archive.read('bundle.manifest.json'))
        if manifest.get('format') != 'baxter-harness-bundle-v1':raise ValueError('Unsupported bundle')
        if manifest.get('kind') not in ('source', 'private-site-config'):raise ValueError('Unsupported bundle kind')
        expected = set(manifest['files']) | {'bundle.manifest.json'}
        if set(names) != expected:raise ValueError('Archive and manifest contain different files')
        for name, value in manifest['files'].items():
            part = PurePosixPath(name)
            if part.is_absolute() or '..' in part.parts or '\\' in name or ':' in name:
                raise ValueError('Unsafe archive path')
            if manifest['kind'] == 'source' and not allowed_source(name):raise ValueError('Non-source file in source bundle')
            if manifest['kind'] == 'private-site-config' and (len(part.parts) != 2 or part.parts[0] != 'config' or not name.endswith('.local.json')):
                raise ValueError('Non-configuration file in private bundle')
            raw = archive.read(name)
            if manifest['kind'] == 'source':validate_source(name, raw)
            if len(raw) != value['bytes'] or hashlib.sha256(raw).hexdigest() != value['sha256']:
                raise ValueError('Archive checksum mismatch')
    return {'verified_files': len(expected) - 1, 'kind': manifest['kind']}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--output', type=Path, default=ROOT / 'dist' / 'baxter-harness-source.zip')
    parser.add_argument('--site-config', type=Path, action='append', default=[])
    parser.add_argument('--site-output', type=Path)
    parser.add_argument('--verify', type=Path, help='Verify an existing archive without extraction or networking')
    args = parser.parse_args(argv)
    try:
        if args.verify:
            print(json.dumps(verify_archive(args.verify), indent=2));return 0
        revision = subprocess.run(['git', '-C', str(args.root), 'rev-parse', 'HEAD'], check=True, capture_output=True,
                                  text=True).stdout.strip()
        source = tracked_sources(args.root)
        result = {'source': write_bundle(args.output, source, 'source', revision)}
        if args.site_config:
            destination = args.site_output or args.output.with_name(args.output.stem + '-private-site.zip')
            if destination.resolve() == args.output.resolve():raise ValueError('Source and private bundles need distinct paths')
            result['private_site'] = write_bundle(destination, site_configs(args.site_config), 'private-site-config', revision)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (OSError, ValueError, subprocess.CalledProcessError, zipfile.BadZipFile) as error:
        print('Bundle failed: ' + type(error).__name__ + ': ' + str(error), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
