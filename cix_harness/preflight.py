"""Offline deployment checks by default; explicit --online adds read-only RPCs."""
import argparse
import hashlib
import html
import importlib.util
import json
from pathlib import Path, PurePosixPath
import sys
import time

from .contract import image_bytes, vector


def read_json(path):
    path = Path(path)
    if path.stat().st_size > 1024 * 1024:
        raise ValueError('JSON input exceeds 1 MiB')
    def finite(_value):
        raise ValueError('JSON contains a nonfinite number')
    return json.loads(path.read_text(encoding='utf-8-sig'), parse_constant=finite)


def verify_manifest(root, manifest_path):
    """Verify extracted file content, refusing paths escaping the release directory."""
    root = Path(root).resolve()
    manifest = read_json(manifest_path)
    if manifest.get('format') != 'baxter-harness-bundle-v1' or not isinstance(manifest.get('files'), dict):
        raise ValueError('Unsupported bundle manifest')
    checked = 0
    for name, expected in manifest['files'].items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or '\\' in name or ':' in name:
            raise ValueError('Manifest path is not a relative release path')
        original = root / name
        path = original.resolve()
        if root not in path.parents or original.is_symlink():
            raise ValueError('Manifest path escapes release directory')
        if not isinstance(expected, dict) or set(expected) != {'sha256', 'bytes'}:
            raise ValueError('Invalid manifest file record')
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != expected['sha256'] or path.stat().st_size != expected['bytes']:
            raise ValueError('Bundle checksum mismatch: ' + name)
        checked += 1
    return {'checked_files': checked, 'bundle_kind': manifest.get('kind'), 'source_revision': manifest.get('source_revision')}


def check_config(path):
    """Report candidate geometry honestly; false commissioning flags stay pending."""
    result = {'file': str(path), 'json_syntax': 'error', 'schema': 'not_checked', 'arms': {}}
    try:
        cfg = read_json(path)
        if not isinstance(cfg, dict):raise ValueError('Configuration must be an object')
        result['json_syntax'] = 'ok'
        if isinstance(cfg.get('arms'), dict):
            from .dual_bridge import validate_dual_config
            validate_dual_config(cfg, armed=False)
            arms = cfg['arms']
            result['inter_arm_clearance_verified'] = (cfg.get('inter_arm_clearance') or {}).get('verified') is True
            result['kind'] = 'dual'
        else:
            from .baxter import validate_config
            validate_config(cfg, armed=False)
            arms = {cfg.get('arm', 'unknown'): cfg}
            result['kind'] = 'single'
        result['schema'] = 'ok'
        for arm, config in arms.items():
            primitive = config.get('primitives', {})
            result['arms'][arm] = {
                'commissioned': config.get('commissioned') is True,
                'primitive_calibrated': primitive.get('calibrated') is True,
                'table_plane_present': isinstance(primitive.get('table_plane'), dict),
                'table_polygon_present': isinstance(primitive.get('table_polygon_base_xy'), list),
                'tcp_offset_present': isinstance(primitive.get('tcp_offset_tool_m'), list),
            }
        result['status'] = 'ok' if (result['arms'] and all(
            item['commissioned'] and item['primitive_calibrated'] for item in result['arms'].values())
            and (result['kind'] != 'dual' or result['inter_arm_clearance_verified'])) else 'pending'
    except (OSError, ValueError, TypeError, KeyError, ImportError, AttributeError) as error:
        result['status'] = 'error'
        result['schema'] = 'error' if result['json_syntax'] == 'ok' else 'not_checked'
        # Fixed exception classes disclose neither file content nor credentials.
        result['error_type'] = type(error).__name__
    result['field_verification_required'] = True
    return result


def _state_check(state):
    if not isinstance(state, dict):return False
    try:
        vector(state['eef_m'], 3, 'endpoint')
        vector(state['eef_xyzw'], 4, 'endpoint orientation')
        vector(state['joint_rad'], 7, 'joint telemetry')
        names = state['joint_names']
        return (isinstance(names, list) and len(names) == 7 and len(set(names)) == 7
                and type(state.get('gripping')) is bool)
    except (KeyError, TypeError, ValueError):
        return False


def save_workspace_view(observation, path):
    """Save an unmodified RGB frame with separate SVG reference-plane overlays."""
    image = observation['image']
    image_bytes(image)  # Reject malformed/oversized data URIs before embedding.
    polygons = []
    labels = []
    for arm, color in (('left', '#00aaff'), ('right', '#ff9f1c')):
        metadata = observation.get('workspaces', {}).get(arm, {})
        reference = metadata.get('table_reference') or {}
        vertices = reference.get('polygon_uv')
        if isinstance(vertices, list) and 3 <= len(vertices) <= 64:
            vertices = [vector(point, 2, 'projected reference vertex') for point in vertices]
            points = ' '.join('{},{}'.format(u, v) for u, v in vertices)
            polygons.append('<polygon points="{}" fill="none" stroke="{}" stroke-width="0.004"/>'.format(points, color))
        labels.append({'arm': arm, 'color': color, 'reference': reference,
                       'calibrated': metadata.get('profile_calibrated'),
                       'note': str(metadata.get('calibration_note', ''))[:300]})
    document = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<title>Baxter 当前桌面参考投影</title><style>
body{font-family:system-ui,sans-serif;max-width:1050px;margin:24px auto;padding:0 16px}
.frame{position:relative}.frame img{display:block;width:100%%;height:auto}
.frame svg{position:absolute;inset:0;width:100%%;height:100%%}pre{white-space:pre-wrap}
</style><h1>当前图像与桌面参考平面投影</h1>
<p>蓝色：机器人左臂；橙色：机器人右臂。原始相机图像未修改，两条边界重合时后者可能遮住前者。</p>
<div class="frame"><img alt="Current unmodified camera frame" src="%s">
<svg viewBox="0 0 1 1" preserveAspectRatio="none" aria-label="Table reference boundaries">%s</svg></div>
<p>边界来自当前配置，并非新的标定证据。核对桌边与固定参考点；高物体存在视差。
多边形内部不等于 IK 可达或无碰撞。无可投影边界时不画猜测线。</p><pre>%s</pre></html>''' % (
        html.escape(image, quote=True), ''.join(polygons), html.escape(json.dumps(labels, ensure_ascii=False, indent=2)))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding='utf-8')


def check_online(robot, view_dir=None):
    """Only fixed, read-only dual RPCs. Do not enable, calibrate, prepare or execute."""
    result = {'status': 'error', 'checks': {}, 'motor_commands_sent': False,
              'ik_preflight': 'pending_task_preview'}
    began = time.monotonic()
    try:
        health = robot.call('healthz', timeout_s=10)
        dual = health.get('dual_protocol') or health.get('protocol') or health.get('hybrid_protocol')
        result['checks']['health'] = {'status': 'ok', 'armed': health.get('armed') is True,
            'motion_scope': health.get('motion_scope'), 'protocol': dual}
        if health.get('motion_scope') != 'supervised_dual_primitive_task':
            result['checks']['dual_bridge'] = {'status': 'error', 'reason': 'wrong_motion_scope'}
            return result
        for arm in ('left', 'right'):
            tick = time.monotonic()
            observation = robot.call('dual.observe', (arm,), timeout_s=20)
            states = observation.get('arm_states', {})
            workspace = observation.get('workspaces', {}).get(arm, {})
            image = observation.get('image')
            camera_ok = isinstance(image, str) and image.startswith(('data:image/jpeg;base64,', 'data:image/png;base64,'))
            states_ok = all(_state_check(states.get(name)) and states[name].get('arm') == name for name in ('left', 'right'))
            selected_ok = observation.get('selected_arm') == arm
            result['checks'][arm] = {'status': 'ok' if camera_ok and states_ok and workspace and selected_ok else 'error',
                'camera_frame_received': camera_ok, 'both_arm_telemetry_valid': states_ok,
                'selected_arm_matches_request': selected_ok,
                'workspace_metadata_present': bool(workspace),
                'seconds': round(time.monotonic() - tick, 3)}
            if view_dir is not None and camera_ok:
                path = Path(view_dir) / (arm + '-workspace.html')
                save_workspace_view(observation, path)
                result['checks'][arm]['view_file'] = str(path)
        clearance = health.get('clearance') or {}
        links = clearance.get('links') or {}
        result['clearance'] = {'configured': clearance.get('configured') is True,
                              'verified': clearance.get('verified') is True,
                              'covered_link_counts': {arm: len(links.get(arm, [])) for arm in ('left', 'right')},
                              'onsite_geometry_check_required': True}
        result['status'] = 'ok' if all(check['status'] == 'ok' for check in result['checks'].values()) else 'error'
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
        result['error_type'] = type(error).__name__
    finally:
        result['seconds'] = round(time.monotonic() - began, 3)
    return result


def report(config_paths=(), credentials=Path('.local/chatgpt-session.json'), manifest=None, root=Path('.'),
           online=False, check_account=False, robot_port=9102, robot=None, view_dir=None):
    result = {'mode': 'online_read_only' if online else 'offline', 'ready_for_motion': False,
              'python': '.'.join(map(str, sys.version_info[:3])), 'configurations': [],
              'field_checks_required': ['fresh camera-to-base and table-plane verification',
                  'right-arm TCP measurement', 'inter-arm geometry and model coverage',
                  'fresh task-specific full IK and trajectory preview', 'on-site motion supervision']}
    result['credentials'] = {'path': str(credentials), 'present': Path(credentials).is_file(),
                             'contents_read': False, 'account_access_verified': False}
    result['auth_dependencies'] = {name: importlib.util.find_spec(name) is not None for name in ('jwt', 'cryptography')}
    result['configurations'] = [check_config(path) for path in config_paths]
    if not config_paths:result['configuration_status'] = 'pending_no_config_selected'
    if manifest:
        try:result['bundle'] = dict(status='ok', **verify_manifest(root, manifest))
        except (OSError, ValueError, TypeError, KeyError) as error:
            result['bundle'] = {'status': 'error', 'error_type': type(error).__name__}
    if check_account and not online:
        raise ValueError('--check-account requires --online; offline never refreshes credentials')
    if view_dir is not None and not online:
        raise ValueError('--view-dir requires --online; offline never captures a camera frame')
    if online:
        if robot is None:
            from .rpc import HttpRpcClient
            robot = HttpRpcClient(robot_port)
        result['robot'] = check_online(robot, view_dir)
        if check_account:
            from .chatgpt_auth import list_models
            try:
                models = list_models(credentials)
                result['credentials'].update(contents_read=True, account_access_verified=True,
                    astra_available=any('astra' in item.get('slug', '').lower() for item in models))
            except (OSError, RuntimeError, ValueError, TypeError, KeyError) as error:
                result['credentials'].update(contents_read=True, error_type=type(error).__name__)
    result['checks_passed'] = not any(item['status'] == 'error' for item in result['configurations'])
    if result.get('bundle', {}).get('status') == 'error' or result.get('robot', {}).get('status') == 'error':
        result['checks_passed'] = False
    if check_account and not result['credentials']['account_access_verified']:result['checks_passed'] = False
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, action='append', default=[])
    parser.add_argument('--credentials', type=Path, default=Path('.local/chatgpt-session.json'))
    parser.add_argument('--manifest', type=Path)
    parser.add_argument('--root', type=Path, default=Path('.'))
    parser.add_argument('--online', action='store_true')
    parser.add_argument('--check-account', action='store_true', help='With --online only: refresh if needed and query model catalog, no inference')
    parser.add_argument('--robot-port', type=int, default=9102)
    parser.add_argument('--view-dir', type=Path, help='With --online only: private local HTML views of current reference-plane projections')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    try:
        result = report(args.config, args.credentials, args.manifest, args.root,
                        args.online, args.check_account, args.robot_port, view_dir=args.view_dir)
        text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + '\n', encoding='utf-8')
        print(text)
        return 0 if result['checks_passed'] else 1
    except (ValueError, OSError) as error:
        parser.exit(1, 'Preflight could not complete: ' + type(error).__name__ + '\n')


if __name__ == '__main__':
    raise SystemExit(main())
