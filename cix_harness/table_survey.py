"""Fit a base-frame tabletop from a fresh perimeter survey and an optical-to-base transform."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from .contract import vector
from .pi05_contract import quaternion, rotate
from .primitives import table_height, validate_table_geometry


def build_geometry(extrinsics, survey, *, max_height_error_m, optical_frame='camera_color_optical_frame',
                   max_reference_error_m=None):
    if not isinstance(optical_frame,str) or not optical_frame.strip() or optical_frame.strip('/')=='base':
        raise ValueError('Specify the optical frame reported by the aligned D455 color stream')
    if (extrinsics['from_frame'], extrinsics['to_frame']) != (optical_frame, 'base'):
        raise ValueError('Expected optical-to-base transform from the selected D455 color frame')
    if extrinsics['units'] != 'm' or survey['units'] != 'm':
        raise ValueError('Transform and survey units must be metres (m)')
    if not all(isinstance(item.get('source'), str) and item['source'].strip() for item in (extrinsics, survey)):
        raise ValueError('Record the calibration and measurement sources')
    if type(max_height_error_m) not in (float, int) or not math.isfinite(max_height_error_m) or max_height_error_m <= 0:
        raise ValueError('Choose a positive maximum allowed table height residual in metres')
    translation = vector(extrinsics['translation_m'], 3, 'camera-to-base translation')
    q = quaternion(extrinsics['xyzw'])

    def to_base(group):
        frame = group['frame']
        if frame not in ('base', optical_frame):
            raise ValueError('Measured points must be in Baxter base or the same D455 color optical frame')
        result = []
        for raw in group['points_m']:
            point = vector(raw, 3, 'survey point')
            if frame == optical_frame:
                if point[2] <= 0:
                    raise ValueError('Optical-frame observation has nonpositive depth')
                point = [x+t for x, t in zip(rotate(q, point), translation)]
            result.append(point)
        return result

    boundary = to_base(survey['boundary'])
    if not 3 <= len(boundary) <= 32:
        raise ValueError('Supply 3 to 32 perimeter points in order')
    corners = np.asarray(boundary)
    centre = corners.mean(axis=0)
    _, singular_values, normals = np.linalg.svd(corners-centre, full_matrices=False)
    if singular_values[1] <= max(1e-6, 1e-8 * singular_values[0]):
        raise ValueError('Degenerate tabletop measurements')
    normal = normals[-1]
    if normal[2] < 0:
        normal = -normal
    plane = {'normal': normal.tolist(), 'offset': -float(normal @ centre)}
    profile = {'table_plane': plane, 'table_polygon_base_xy': [p[:2] for p in boundary],
               'tcp_workspace_min_m': corners.min(axis=0).tolist()[:2]+[-2.],
               'tcp_workspace_max_m': corners.max(axis=0).tolist()[:2]+[2.]}
    validate_table_geometry(profile)
    check_points = to_base(survey['check_points']) if 'check_points' in survey else []
    pairs = survey.get('reference_pairs', [])
    reference_errors = []
    if pairs:
        if type(max_reference_error_m) not in (float, int) or not math.isfinite(max_reference_error_m) or max_reference_error_m <= 0:
            raise ValueError('Choose a positive 3-D reference error tolerance for paired measurements')
        for pair in pairs:
            optical = to_base({'frame': optical_frame, 'points_m': [pair['color_optical_m']]})[0]
            reference = vector(pair['measured_base_m'], 3, 'independently measured base reference')
            reference_errors.append(math.dist(optical, reference))
        if max(reference_errors) > max_reference_error_m:
            raise ValueError('Optical-to-base 3-D reference error {:.6f} m exceeds {:.6f} m'.format(
                max(reference_errors), max_reference_error_m))

    def residuals(points):
        errors = [z-table_height(x, y, profile) for x, y, z in points]
        return {'count': len(points), 'signed_z_errors_m': errors,
                'rms_m': math.sqrt(sum(e*e for e in errors)/len(errors)) if errors else None,
                'max_abs_m': max(map(abs, errors), default=None)}

    fit = residuals(boundary)
    independent = residuals(check_points)
    for label, evidence in [('boundary', fit), ('independent check', independent)]:
        if evidence['count'] and evidence['max_abs_m'] > max_height_error_m:
            raise ValueError('{} vertical residual {:.6f} m exceeds {:.6f} m'.format(
                label, evidence['max_abs_m'], max_height_error_m))
    return {
        'status': 'table_survey_candidate', 'commissioned': False, 'calibrated': False,
        'source': survey['source'] + ' / ' + extrinsics['source'],
        'camera_to_base_xyzw': q, 'camera_to_base_translation_m': translation,
        'interpreted_boundary_base_m': boundary, 'plane': plane,
        'fit': fit, 'independent_check': independent, 'check_points_base_m': check_points,
        'optical_frame': optical_frame, 'boundary_input_frame': survey['boundary']['frame'],
        'check_input_frame': survey.get('check_points', {}).get('frame'),
        'max_height_error_m': max_height_error_m,
        'reference_errors_m': reference_errors, 'max_reference_error_m': max_reference_error_m,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--extrinsics', type=Path, required=True)
    parser.add_argument('--survey', type=Path, required=True)
    parser.add_argument('--max-height-error-m', type=float, required=True)
    parser.add_argument('--max-reference-error-m', type=float, help='3-D tolerance for optional same-point pairs')
    parser.add_argument('--optical-frame', default='camera_color_optical_frame')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(argv)
    raw = [path.read_bytes() for path in (args.extrinsics, args.survey)]
    geometry = build_geometry(*(json.loads(data.decode('utf-8-sig')) for data in raw),
                              max_height_error_m=args.max_height_error_m, optical_frame=args.optical_frame,
                              max_reference_error_m=args.max_reference_error_m)
    geometry['inputs'] = {key: {'file': path.name, 'sha256': hashlib.sha256(data).hexdigest()}
                          for key,path,data in zip(('extrinsics','survey'), (args.extrinsics,args.survey), raw)}
    encoded = json.dumps(geometry, ensure_ascii=False, indent=2, allow_nan=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8') as output:
        output.write(encoded+'\n')
    print(json.dumps({'output': str(args.output), 'status': geometry['status'],
                      'fit_max_abs_m': geometry['fit']['max_abs_m'],
                      'independent_check_max_abs_m': geometry['independent_check']['max_abs_m']}))


if __name__ == '__main__':
    main()
