"""Design a spatial hand-eye calibration and audit the measured marker distribution."""
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path

import numpy as np

from .handeye import angle, load, packed, pose_arrays, rotation_vector, solve, transform


GRID=np.array([3,2,2])
CELLS=list(itertools.product(range(3),range(2),range(2)))
HOLD_CELLS=[c for c in CELLS if sum(c)%2==0]
MIN_SPAN_MM={'train':np.array([180.,160.,100.]),'holdout':np.array([100.,70.,45.])}


def bounds_arrays(bounds):
    if bounds.get('frame')!='base' or bounds.get('units')!='m':
        raise ValueError('Marker bounds require Baxter base frame and metres')
    lo,hi=(np.asarray(bounds[n],dtype=float) for n in ('lower','upper'))
    if lo.shape!=(3,) or hi.shape!=(3,) or not np.isfinite([lo,hi]).all() or np.any(hi<=lo):
        raise ValueError('Expected finite increasing 3D marker workspace bounds')
    return lo,hi


def cell_index(point,lo,hi):
    ratio=(point-lo)/(hi-lo)
    if not np.isfinite(ratio).all() or min(ratio)<0 or max(ratio)>1:return None
    return tuple(np.minimum(GRID-1,np.floor(ratio*GRID).astype(int)).tolist())


def generate_plan(bounds,reference):
    lo,hi=bounds_arrays(bounds)
    view=transform(reference['base_T_effector'])
    offset=(transform(reference['effector_T_tag'])[:3,3] if 'effector_T_tag' in reference else
            np.asarray(reference['effector_tag_center_m'],dtype=float))
    if offset.shape!=(3,) or not np.isfinite(offset).all():raise ValueError('Invalid wrist-to-marker centre offset')
    cells=sorted(CELLS,key=lambda c:(c[2],c[1],c[0] if c[1]%2==0 else -c[0]))
    poses=[]
    # Two distinct locations per voxel plus a preassigned independent middle
    # location in half of them. Alternate the signs so positions fill the volume.
    for k,cell in enumerate(cells):
        flip=np.array([1 if (k//2)%2 else -1,1 if k%2 else -1,1 if (k//3)%2 else -1])
        for slot in ((0,2,1) if cell in HOLD_CELLS else (0,1)):
            fraction=.5+(.27*flip if slot==0 else -.27*flip if slot==1 else np.zeros(3))
            center=lo+(np.array(cell)+fraction)/GRID*(hi-lo)
            phase=math.radians((k*137+slot*107)%360)
            rv=np.radians([17*math.cos(phase),17*math.sin(phase),(-1)**(k+slot)*12])
            theta=np.linalg.norm(rv)
            delta=transform({'xyz':[0.,0.,0.],'xyzw':(rv*math.sin(theta/2)/theta).tolist()+[math.cos(theta/2)]})
            wrist=view@delta
            wrist[:3,3]=center-wrist[:3,:3]@offset
            poses.append({'id':'v{:02d}'.format(len(poses)+1),'split':'holdout' if slot==2 else 'train',
                          'cell':list(cell),'tag_center_base_m':center.tolist(),'base_T_effector':packed(wrist)})
    plan={'schema':'handeye-volume-v1','bounds':bounds,'reference':reference,'poses':poses}
    validate_plan(plan)
    return plan


def spread(points):
    if not len(points):return {'span_mm':[0.]*3,'pca_rms_mm':[0.]*3}
    points=np.asarray(points)*1000.
    singular=np.linalg.svd(points-points.mean(axis=0),compute_uv=False)/math.sqrt(len(points))
    return {'span_mm':np.ptp(points,axis=0).tolist(),'pca_rms_mm':np.pad(singular,(0,3-len(singular))).tolist()}


def angle_excitation(rotations):
    rv=[]
    for a,b in itertools.combinations(rotations,2):
        r=b@a.T;t=angle(r)
        if t>1e-8:rv.append(rotation_vector(r,t))
    if len(rv)<2:return 0.
    return math.degrees(np.linalg.svd(rv,compute_uv=False)[1]/math.sqrt(len(rv)))


def validate_plan(plan):
    if plan.get('schema')!='handeye-volume-v1':raise ValueError('Wrong spatial calibration plan schema')
    lo,hi=bounds_arrays(plan['bounds']);seen=set();counts={};points=[]
    for p in plan['poses']:
        id_=p['id'];s=p['split'];c=tuple(p['cell'])
        if not isinstance(id_,str) or not id_ or id_ in seen:raise ValueError('Duplicate or missing plan ID')
        seen.add(id_);center=np.asarray(p['tag_center_base_m'],dtype=float);points.append(center)
        if center.shape!=(3,) or cell_index(center,lo,hi)!=c:raise ValueError('Planned marker centre outside assigned cell')
        if s not in MIN_SPAN_MM:raise ValueError('Invalid preassigned split')
        counts[c,s]=counts.get((c,s),0)+1
        transform(p['base_T_effector'])
    for c in CELLS:
        if counts.get((c,'train'),0)<2 or (c in HOLD_CELLS and counts.get((c,'holdout'),0)<1):
            raise ValueError('Incomplete 24 training + 6 independent holdout volume layout')
    for s,mins in MIN_SPAN_MM.items():
        chosen=[p for p in plan['poses'] if p['split']==s]
        for name,group_points in [('marker',[p['tag_center_base_m'] for p in chosen]),
                                  ('wrist',[transform(p['base_T_effector'])[:3,3] for p in chosen])]:
            if np.any(np.asarray(spread(group_points)['span_mm'])+1e-6<mins):
                raise ValueError(f'Planned {s} {name} bounds too small for {mins.tolist()} mm minimum spans')
    if min(np.linalg.norm(a-b) for a,b in itertools.combinations(points,2))<.025:
        raise ValueError('Planned marker centres must be at least 25 mm apart')
    wrists=[transform(p['base_T_effector']) for p in plan['poses']]
    for a,b in itertools.combinations(wrists,2):
        # Each capture can be 5 mm and 2 degrees from its target. The actual
        # recorder checks 35 mm OR 12 degrees between measured wrist poses.
        if np.linalg.norm(a[:3,3]-b[:3,3])<.045 and angle(a[:3,:3].T@b[:3,:3])<math.radians(16.):
            raise ValueError('Two planned wrist poses may collide with the synchronized recorder novelty gate')


def audit_volume(plan,fit,holdout,camera,manifest):
    validate_plan(plan)
    by_id={p['id']:p for p in plan['poses']}
    lo,hi=bounds_arrays(plan['bounds']);X=transform(camera)
    intr=manifest.get('intrinsics',{});size=np.asarray([intr.get('width',0),intr.get('height',0)],dtype=float)
    if not np.isfinite(size).all() or min(size)<=1:raise ValueError('Missing actual color image dimensions in manifest')
    errors=[];groups={};allpoints=[];all_ids=set()
    for split,rows in [('train',fit),('holdout',holdout)]:
        arms=[];opt=[];centers=[];uv=[];rots=[];cells={};per_cell_rot={};details=[]
        for obs in rows:
            key=obs.get('drive_pose_id')
            if key not in by_id or key in all_ids or by_id[key]['split']!=split:
                errors.append(f'{split}: unplanned, repeated or misassigned pose ID {key}');continue
            all_ids.add(key);p=by_id[key]
            if manifest.get('volume_plan_sha256') and (
                    obs.get('volume_plan_sha256')!=manifest['volume_plan_sha256'] or
                    obs.get('volume_split')!=split or obs.get('volume_cell')!=p['cell']):
                errors.append(key+' captured plan hash, split or cell disagrees with the frozen assignment')
            E=transform(obs['base_T_effector']);C=transform(obs['camera_T_tag'])
            B=X@C;center=B[:3,3];cell=cell_index(center,lo,hi)
            corners=np.asarray(obs.get('corners_px'),dtype=float)
            if corners.shape!=(4,2) or not np.isfinite(corners).all() or np.any(corners<0) or np.any(corners>=size):
                errors.append(key+' invalid measured image corners');continue
            pixel=corners.mean(axis=0)/size
            if cell!=tuple(p['cell']):errors.append(key+' measured tag outside assigned volume cell')
            goal=transform(p['base_T_effector'])
            pos_err=np.linalg.norm(E[:3,3]-goal[:3,3])*1000
            ang_err=math.degrees(angle(E[:3,:3].T@goal[:3,:3]))
            if pos_err>5.01 or ang_err>2.01:errors.append(key+' measured wrist outside assigned pose gate')
            details.append({'id':key,'cell':cell,'tag_base_m':center.tolist(),'wrist_base_m':E[:3,3].tolist(),
                            'tag_center_uv_normalized':pixel.tolist(),'target_wrist_error_mm':pos_err,'target_wrist_error_deg':ang_err})
            cells[cell]=cells.get(cell,0)+1
            per_cell_rot.setdefault(cell,[]).append(C[:3,:3]);rots.append(C[:3,:3])
            centers.append(center);opt.append(C[:3,3]);arms.append(E[:3,3]);uv.append(pixel);allpoints.append((key,center))
        stats=spread(centers);wrist_stats=spread(arms);exc=angle_excitation(rots)
        if np.any(np.asarray(stats['span_mm'])<MIN_SPAN_MM[split]):errors.append(split+' measured marker base span too small')
        if np.any(np.asarray(wrist_stats['span_mm'])<MIN_SPAN_MM[split]):errors.append(split+' measured wrist base span too small')
        if spread(opt)['pca_rms_mm'][2]<(20. if split=='train' else 12.):errors.append(split+' observed optical marker cloud lacks volume')
        if exc<(8. if split=='train' else 6.):errors.append(split+' observed tag rotation lacks multiple axes')
        for c in (CELLS if split=='train' else HOLD_CELLS):
            needed=2 if split=='train' else 1
            if cells.get(c,0)<needed:errors.append(f'{split} insufficient measured markers in {c}')
            if split=='train' and cells.get(c,0)>=2:
                largest=max(math.degrees(angle(a.T@b)) for a,b in itertools.combinations(per_cell_rot[c],2))
                if largest<10:errors.append(f'{split} local views differ by less than 10 degrees in {c}')
        uv_span=np.ptp(uv,axis=0) if uv else np.zeros(2)
        if np.min(uv_span)<(.06 if split=='train' else .035):errors.append(split+' marker image centres occupy too small a region')
        groups[split]={'count':len(details),'tag_base':stats,'tag_optical':spread(opt),'wrist_base':wrist_stats,
                       'rotation_second_axis_rms_deg':exc,'cell_counts':{str(k):v for k,v in cells.items()},
                       'center_uv_min':np.min(uv,axis=0).tolist() if uv else None,
                       'center_uv_max':np.max(uv,axis=0).tolist() if uv else None,'center_uv_span':uv_span.tolist(),'samples':details}
    missing=sorted(set(by_id)-all_ids)
    if missing:errors.append('Missing planned pose IDs: '+', '.join(missing))
    closest=min(((float(np.linalg.norm(a-b)*1000),i,j) for (i,a),(j,b) in itertools.combinations(allpoints,2)),default=None)
    if closest is None or closest[0]<25:errors.append('Two measured tag centres are less than 25 mm apart or missing')
    return {'passed':not errors,'failures':errors,'groups':groups,'closest_tag_pair_mm_ids':closest,
            'minimum_base_span_mm':{s:v.tolist() for s,v in MIN_SPAN_MM.items()}}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);subs=p.add_subparsers(dest='command',required=True)
    make=subs.add_parser('plan');audit=subs.add_parser('audit')
    for cli,names in ((make,('bounds','reference','output')),(audit,('plan','fit','holdout','manifest','output'))):
        for n in names:cli.add_argument('--'+n,required=True,type=Path)
    args=p.parse_args(argv)
    if args.command=='plan':
        result=generate_plan(load(args.bounds),load(args.reference))
    else:
        fit,hold=load(args.fit),load(args.holdout)
        fit_result=solve(pose_arrays(fit),pose_arrays(hold),10,3)
        manifest=load(args.manifest)
        result=audit_volume(load(args.plan),fit,hold,fit_result['base_T_camera'],manifest)
        result['volume_plan_sha256']=hashlib.sha256(args.plan.read_bytes()).hexdigest()
        if manifest.get('volume_plan_sha256')!=result['volume_plan_sha256']:
            result['failures'].append('Capture manifest does not attest to the supplied volume plan hash')
            result['passed']=False
    with args.output.open('x',encoding='utf8') as stream:json.dump(result,stream,ensure_ascii=False,indent=2,allow_nan=False)
    return 0 if result.get('passed',True) else 2


if __name__=='__main__':raise SystemExit(main())
