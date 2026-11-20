"""Solve a fixed external camera and rigid wrist tag from synchronized, stationary poses."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from .contract import vector
from .pi05_contract import quaternion, rotate


def transform(value):
    if isinstance(value, dict):
        xyz = vector(value['xyz'], 3, 'translation in metres')
        q = quaternion(value['xyzw'])
        m = np.eye(4)
        m[:3, :3] = np.column_stack([rotate(q, axis) for axis in np.eye(3)])
        m[:3, 3] = xyz
    else:
        m = np.asarray(value, dtype=float)
    if m.shape != (4, 4) or not np.isfinite(m).all() or not np.allclose(m[3], [0,0,0,1], atol=1e-7):
        raise ValueError('Expected a finite homogeneous transform')
    r = m[:3, :3]
    if not np.allclose(r.T @ r, np.eye(3), atol=2e-4) or abs(np.linalg.det(r) - 1.) > 2e-4:
        raise ValueError('Expected a right-handed orthonormal rotation')
    return m


def pose_arrays(data):
    if isinstance(data, dict):
        # easy_handeye's ROS1 get_sample_list for eye-on-base records
        # E<-base (hand_world), while the optical half is camera<-tag.
        samples = data.get('samples', data.get('sample_list', data))
        if isinstance(samples, dict):
            arms = samples['hand_world_samples']
            tags = samples['camera_marker_samples']
            if len(arms) != len(tags):
                raise ValueError('easy_handeye sample lists have different lengths')
            def ros_pose(t):
                return transform({'xyz':[t['translation'][k] for k in 'xyz'],
                                  'xyzw':[t['rotation'][k] for k in 'xyzw']})
            return [(np.linalg.inv(ros_pose(a)), ros_pose(c)) for a,c in zip(arms,tags)]
        data = samples
    return [(transform(s['base_T_effector']), transform(s['camera_T_tag'])) for s in data]


def angle(r):
    return math.acos(float(np.clip((np.trace(r)-1.)/2., -1., 1.)))


def rotation_vector(r, theta):
    return np.array([r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]]) * (theta/(2.*math.sin(theta)))


def closest_rotation(matrix):
    u, _, vt = np.linalg.svd(matrix)
    sign = 1. if np.linalg.det(u @ vt) > 0 else -1.
    return u @ np.diag([1.,1.,sign]) @ vt


def packed(m):
    r = m[:3,:3]
    k = np.zeros((4,4))
    k[:3,:3] = r+r.T-np.eye(3)*np.trace(r)
    k[3,3] = np.trace(r)
    k[:3,3] = k[3,:3] = [r[2,1]-r[1,2],r[0,2]-r[2,0],r[1,0]-r[0,1]]
    q = np.linalg.eigh(k)[1][:,-1]
    if q[3]<0: q=-q
    return {'xyz':m[:3,3].tolist(),'xyzw':q.tolist(),'matrix':m.tolist()}


def tool_pose(base_effector, base_camera, camera_tag):
    return np.linalg.solve(base_effector, base_camera @ camera_tag)


def check_distinct(fit, holdout):
    seen = []
    for group, pairs in [('fit',fit),('holdout',holdout)]:
        for i,(arm,_) in enumerate(pairs):
            for prev_group,j,prev in seen:
                if (np.linalg.norm(arm[:3,3]-prev[:3,3]) < .001 and
                        angle(arm[:3,:3].T @ prev[:3,:3]) < math.radians(.5)):
                    raise ValueError(f'Duplicate pose {prev_group}[{j}] and {group}[{i}]; supply distinct stationary poses')
            seen.append((group,i,arm))


def solve(fit, holdout, max_mm, max_deg, known_camera=None, min_excitation_deg=3.):
    """For each pose, B_T_E @ E_T_tag must equal B_T_camera @ C_T_tag."""
    if len(fit) < 3 or len(holdout) < 3:
        raise ValueError('Supply at least three training and three distinct held-out wrist poses')
    if any(not math.isfinite(v) or v<=0 for v in (max_mm,max_deg,min_excitation_deg)):
        raise ValueError('Expected positive finite error limits and excitation')
    check_distinct(fit, holdout)
    axes_l, axes_r, rel_l, rel_r = [], [], [], []
    for (ei,ci),(ej,cj) in itertools.combinations(fit, 2):
        # Eliminate the fixed wrist tag to obtain L X = X R with X=B_T_camera.
        l, r = ej @ np.linalg.inv(ei), cj @ np.linalg.inv(ci)
        al, ar = angle(l[:3,:3]), angle(r[:3,:3])
        if min(al,ar)<1e-6 or max(al,ar)>math.radians(175):
            continue
        axes_l.append(rotation_vector(l[:3,:3],al))
        axes_r.append(rotation_vector(r[:3,:3],ar))
        rel_l.append(l)
        rel_r.append(r)
    if len(axes_l)<2:
        raise ValueError('Insufficient relative rotations for hand-eye calibration')
    axes_l, axes_r = np.asarray(axes_l), np.asarray(axes_r)
    excitation = min(np.linalg.svd(a,compute_uv=False)[1] for a in (axes_l,axes_r))/math.sqrt(len(axes_l))
    excitation_deg = math.degrees(excitation)
    if excitation_deg < min_excitation_deg:
        raise ValueError(f'Insufficient independent rotation axes ({excitation_deg:.3f} deg RMS)')
    if known_camera is None:
        base_camera = np.eye(4)
        base_camera[:3,:3] = closest_rotation(axes_l.T @ axes_r)
        a = np.vstack([l[:3,:3]-np.eye(3) for l in rel_l])
        b = np.concatenate([base_camera[:3,:3] @ r[:3,3]-l[:3,3] for l,r in zip(rel_l,rel_r)])
        t,_,rank,_ = np.linalg.lstsq(a,b,rcond=None)
        if rank!=3: raise ValueError('Unobservable camera translation')
        base_camera[:3,3] = t
    else:
        base_camera = transform(known_camera)
    tools = [tool_pose(e,base_camera,c) for e,c in fit]
    effector_tag = np.eye(4)
    effector_tag[:3,:3] = closest_rotation(sum(m[:3,:3] for m in tools))
    effector_tag[:3,3] = np.mean([m[:3,3] for m in tools],axis=0)
    def residuals(pairs):
        errors = []
        for i,(e,c) in enumerate(pairs):
            actual, expected = base_camera @ c, e @ effector_tag
            errors.append({'sample_index':i,
                           'mm':float(np.linalg.norm(actual[:3,3]-expected[:3,3])*1000.),
                           'deg':math.degrees(angle(actual[:3,:3].T @ expected[:3,:3]))})
        return {'count':len(errors),'samples':errors,'max_mm':max(x['mm'] for x in errors),
                'rms_mm':math.sqrt(sum(x['mm']**2 for x in errors)/len(errors)),
                'max_deg':max(x['deg'] for x in errors),
                'rms_deg':math.sqrt(sum(x['deg']**2 for x in errors)/len(errors))}
    training, validation = residuals(fit), residuals(holdout)
    return {'fit':training,'holdout':validation,'fit_second_axis_rms_deg':excitation_deg,
            'mode':'verify_existing' if known_camera is not None else 'solve',
            'base_T_camera':packed(base_camera),'effector_T_tag':packed(effector_tag),
            'limits':{'max_mm':max_mm,'max_deg':max_deg,'min_excitation_deg':min_excitation_deg},
            'consistency_passed':all(g['max_mm']<=max_mm and g['max_deg']<=max_deg for g in (training,validation)),
            'commissioned':False}


def load(path):
    if path.suffix.lower() in ('.yaml','.yml'):
        import yaml
        return yaml.safe_load(path.read_bytes())
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('fit','holdout','output'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--camera',type=Path,help='Check an existing easy_handeye eye-on-base YAML, or a base_T_camera pose JSON')
    p.add_argument('--extrinsics-output',type=Path,help='Export the candidate after rigid-pose and formal spatial checks pass')
    p.add_argument('--volume-plan',type=Path,help='Frozen marker-volume plan with at least 24 training and 6 holdout positions')
    p.add_argument('--pose-estimator',choices=('last-frame','stationary-window'),default='last-frame',
                   help='Use recorded endpoint poses or estimate one measured-ruler pose per synchronized RGB-D window')
    p.add_argument('--manifest',type=Path,help='Capture manifest including the plan hash and actual color intrinsics')
    p.add_argument('--optical-frame',help='D455 aligned color CameraInfo frame for the exported transform')
    p.add_argument('--max-mm',required=True,type=float)
    p.add_argument('--max-deg',required=True,type=float)
    p.add_argument('--min-excitation-deg',type=float,default=3.)
    p.add_argument('--holdout-includes-fit',action='store_true',help='easy_handeye cumulative snapshot: verify and remove exact training prefix')
    args=p.parse_args(argv)
    if args.extrinsics_output and (not args.optical_frame or args.optical_frame.strip('/')=='base'):
        p.error('--extrinsics-output requires the observed --optical-frame')
    if args.extrinsics_output and not args.volume_plan:
        p.error('--extrinsics-output requires a preassigned --volume-plan and its capture manifest')
    if args.holdout_includes_fit and (args.volume_plan or args.pose_estimator=='stationary-window'):
        p.error('Synchronized/volume captures store separate explicitly assigned fit and holdout records')
    camera=None
    if args.camera:
        raw=load(args.camera)
        if 'parameters' in raw:
            if raw['parameters'].get('eye_on_hand') is not False or raw['parameters'].get('robot_base_frame','').lstrip('/')!='base':
                p.error('Expected an eye-on-base calibration in the Baxter base frame')
            if args.optical_frame and raw['parameters'].get('tracking_base_frame','').strip('/')!=args.optical_frame.strip('/'):
                p.error('Candidate optical frame differs from input calibration')
            t=raw['transformation']
            camera={'xyz':[t[k] for k in 'xyz'],'xyzw':[t['q'+k] for k in 'xyzw']}
        elif 'camera_to_base_translation_m' in raw:
            if raw.get('target_frame','base').strip('/')!='base':
                p.error('Candidate destination must be Baxter base')
            if args.optical_frame and raw.get('source_frame', '').strip('/')!=args.optical_frame.strip('/'):
                p.error('Candidate source frame differs from supplied optical frame')
            camera={'xyz':raw['camera_to_base_translation_m'],'xyzw':raw['camera_to_base_xyzw']}
        else:
            camera=raw.get('base_T_camera',raw)
    from .handeye_window import estimate_rows
    raw_fit,raw_holdout=load(args.fit),load(args.holdout)
    fit_data,fit_windows=estimate_rows(raw_fit,args.pose_estimator)
    holdout_data,holdout_windows=estimate_rows(raw_holdout,args.pose_estimator)
    fit,holdout=pose_arrays(fit_data),pose_arrays(holdout_data)
    if args.holdout_includes_fit:
        if len(holdout)<=len(fit) or not all(np.allclose(a,b,rtol=0.,atol=1e-10) for a,b in zip(fit,holdout)):
            p.error('Cumulative snapshot must start with the complete, identical fit set and include new poses')
        holdout=holdout[len(fit):]
    report=solve(fit,holdout,args.max_mm,args.max_deg,camera,args.min_excitation_deg)
    report['pose_estimator']=args.pose_estimator
    if fit_windows is not None:
        report['stationary_windows']={'fit':fit_windows,'holdout':holdout_windows}
        try:
            baseline=solve(pose_arrays(raw_fit),pose_arrays(raw_holdout),args.max_mm,args.max_deg,camera,args.min_excitation_deg)
            report['last_frame_diagnostic']={k:baseline[k] for k in ('fit','holdout','base_T_camera','effector_T_tag','consistency_passed')}
        except ValueError as error:
            report['last_frame_diagnostic']={'error':str(error)}
    inputs={'fit':args.fit,'holdout':args.holdout}
    if args.camera:inputs['camera']=args.camera
    if args.volume_plan:
        from .handeye_volume import audit_volume
        manifest_path=args.manifest or args.fit.with_name('manifest.json')
        manifest=load(manifest_path)
        coverage=audit_volume(load(args.volume_plan),fit_data,holdout_data,report['base_T_camera'],manifest)
        plan_digest=hashlib.sha256(args.volume_plan.read_bytes()).hexdigest()
        if manifest.get('volume_plan_sha256')!=plan_digest:
            coverage['failures'].append('Capture manifest does not attest to the supplied volume plan hash')
            coverage['passed']=False
        report['workspace_coverage']=coverage
        inputs.update(volume_plan=args.volume_plan,capture_manifest=manifest_path)
    passed=report['consistency_passed'] and report.get('workspace_coverage',{}).get('passed',True)
    report['accepted']=passed
    report['input_sha256']={key:hashlib.sha256(path.read_bytes()).hexdigest() for key,path in inputs.items()}
    with args.output.open('x',encoding='utf8') as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2,allow_nan=False)
    if passed and args.extrinsics_output:
        pose=report['base_T_camera']
        survey={'from_frame':args.optical_frame,'to_frame':'base','units':'m',
                'translation_m':pose['xyz'],'xyzw':pose['xyzw'],'source':args.output.name,
                'input_sha256':report['input_sha256'],'commissioned':False}
        with args.extrinsics_output.open('x',encoding='utf8') as stream:
            json.dump(survey,stream,ensure_ascii=False,indent=2,allow_nan=False)
    print(json.dumps({'consistency_passed':report['consistency_passed'],'accepted':passed,
                      'holdout_max_mm':report['holdout']['max_mm'],'holdout_max_deg':report['holdout']['max_deg']}))
    return 0 if passed else 2


if __name__=='__main__':raise SystemExit(main())
