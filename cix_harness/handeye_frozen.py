"""Evaluate a previously frozen edge-refit camera AND mount on independent RGB-D captures."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np

from .handeye import angle, transform
from .handeye_capture import duplicate_observation, pose_difference, window_status


def load_record(path, fingerprints):
    raw=path.read_bytes()
    fingerprints[str(path)]=hashlib.sha256(raw).hexdigest()
    return json.loads(raw.decode('utf-8-sig'))


def capture_to_ippe(detector, side):
    """Map centered IPPE object coordinates to the capture's decoded-corner axes."""
    expected=.5*side*np.array([[-1,1,0],[1,1,0],[1,-1,0],[-1,-1,0]])
    points=np.asarray(detector['object_corner_order_m'],dtype=float)
    if points.shape!=(4,3) or not np.allclose(points,expected,rtol=0,atol=1e-7):
        raise ValueError('Historical object points do not have the declared centered IPPE square convention')
    return np.diag([1.,-1.,-1.,1.])


def evaluate(hypothesis, provenance, detector, folders, first_day, max_mm, max_deg):
    if not folders or any(not math.isfinite(v) or v<=0 for v in (max_mm,max_deg)):
        raise ValueError('Provide capture directories and positive finite limits')
    fingerprints={}
    old=load_record(hypothesis,fingerprints)
    source=load_record(provenance,fingerprints)
    historical=source['per_sample']
    if (old['fit_data']!='original training only' or old['training_count']!=len(historical) or
            len(source['matched_detector_frames'])!=len(historical) or len(historical)<3):
        raise ValueError('Historical frozen fit and pose provenance disagree')
    start=datetime.strptime(first_day,'%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()
    if max(float(row['image_stamp']) for row in historical)>=start:
        raise ValueError('Requested new sample day overlaps the historical training period')
    training=[]
    for row,frame in zip(historical,source['matched_detector_frames']):
        if row['image_stamp']!=frame['image_stamp'] or 'DICT_APRILTAG_36h11' not in frame['detector']:
            raise ValueError('Historical detector provenance mismatch')
        training.append(transform({'xyz':row['robot_tf_translation_m'],'xyzw':row['robot_tf_xyzw']}))
    side=float(old['tag_side_m'])
    if not math.isfinite(side) or side<=0:raise ValueError('Invalid historical tag size')
    basis=capture_to_ippe(load_record(detector,fingerprints),side)
    camera=transform(old['edge_refined']['T_base_camera'])
    mount=transform(old['edge_refined']['T_hand_tag'])
    measurements=[];distinct=[];distinct_keys=[];stamps=set();serial=None
    for folder in folders:
        manifest=load_record(folder/'manifest.json',fingerprints)
        if (manifest['arm']!='right' or manifest['effector']!='right_gripper' or manifest['frame']!='base' or
                manifest['tag_family']!='tag36h11' or manifest['tag_id']!=0 or manifest['camera_frame']!='color optical' or
                manifest['tag_axes']!='canonical decoded corners 0->1 x, 0->3 y, x cross y z'):
            raise ValueError('Unrecognized input arm, camera frame, marker or capture convention')
        if not manifest['camera_serial'] or (serial is not None and manifest['camera_serial']!=serial):
            raise ValueError('Different camera serials in independent captures')
        serial=manifest['camera_serial']
        for split in ('train','holdout'):
            for sample in load_record(folder/(split+'.json'),fingerprints):
                # Both splits are unseen by this frozen historical fit.
                stamp=sample['endpoint_ros_stamp_s']
                if not math.isfinite(stamp) or stamp<start or stamp in stamps:
                    raise ValueError('Old or duplicated ROS endpoint sample')
                stamps.add(stamp)
                if (sample['arm']!='right' or sample['tag_family']!='tag36h11' or sample['tag_id']!=0 or
                        sample['camera_serial']!=serial):
                    raise ValueError('Sample identity differs from the capture manifest')
                for key in ('width_m','height_m'):
                    reference=sample['ruler_reference'][key]
                    if (reference is None or not math.isfinite(reference) or abs(reference-side)>max(.005,.04*side)
                            or not math.isfinite(sample[key]) or abs(sample[key]-reference)>max(.005,.04*reference)):
                        raise ValueError('New ruler reference or observed dimensions disagree with historical tag size')
                window=sample['stationary_frames']
                status=window_status(window)
                if status is not None:raise ValueError('Invalid captured stationary window: '+status)
                if window[-1]['endpoint_ros_stamp_s']!=stamp:
                    raise ValueError('Stationary window and saved pose do not have the same endpoint stamp')
                arm=transform(sample['base_T_effector'])
                raw=transform(sample['camera_T_tag'])
                for name,pose in (('base_T_effector',arm),('camera_T_tag',raw)):
                    if not np.allclose(transform(window[-1][name]),pose,rtol=0,atol=1e-8):
                        raise ValueError('Saved pose differs from the stationary frame: '+name)
                key=str(folder/sample['frame_id'])
                predicted=arm@mount
                observed=camera@raw@basis
                dist,rotation=pose_difference(predicted,observed)
                overlap=next((i for i,t in enumerate(training,1)
                              if all(v<lim for v,lim in zip(pose_difference(t,arm),(.001,math.radians(.5))))),None)
                near=duplicate_observation(arm,distinct)
                duplicate_key=distinct_keys[near['nearest_pose_index']] if near else None
                if overlap is None and near is None:
                    distinct.append(arm);distinct_keys.append(key)
                measurements.append({'sample':key,'source_split':split,'stamp_s':stamp,'stationary_frames':len(window),
                    'overlap_historical_pose':overlap,'nearby_new_pose':duplicate_key,
                    'mm':dist*1000,'deg':math.degrees(rotation),
                    'observed_minus_predicted_base_mm':((observed[:3,3]-predicted[:3,3])*1000).tolist(),
                    'deg_if_axes_were_mixed':math.degrees(angle(predicted[:3,:3].T@(camera@raw)[:3,:3]))})
    peak_mm=max((m['mm'] for m in measurements),default=None)
    peak_deg=max((m['deg'] for m in measurements),default=None)
    within_limits=bool(measurements) and peak_mm<=max_mm and peak_deg<=max_deg
    return {'mode':'historical_camera_and_mount_frozen','fit_sources':{'hypothesis':str(hypothesis),
            'pose_provenance':str(provenance),'historical_detector_config':str(detector),'fit_count':len(training)},
            'current_samples_used_for_fit':0,'input_sha256':fingerprints,'capture_camera_serial':serial,
            'first_new_utc_day':first_day,'capture_tag_T_historical_tag':basis.tolist(),'samples':measurements,
            'distinct_new_poses':len(distinct),'minimum_distinct':3,'max_mm':peak_mm,'max_deg':peak_deg,
            'limits':{'max_mm':max_mm,'max_deg':max_deg},'all_observations_within_limits':within_limits,
            'consistency_passed':len(distinct)>=3 and within_limits,
            'status':('insufficient_distinct_poses' if len(distinct)<3 else 'passed' if within_limits else 'outside_limits'),
            'commissioned':False}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('hypothesis','fit-provenance','detector-config','output'):
        p.add_argument('--'+flag,type=Path,required=True)
    p.add_argument('--capture-dir',type=Path,action='append',required=True)
    p.add_argument('--first-new-utc-day',required=True,help='Inclusive YYYY-MM-DD after the historical fit')
    p.add_argument('--max-mm',type=float,required=True)
    p.add_argument('--max-deg',type=float,required=True)
    args=p.parse_args(argv)
    try:
        result=evaluate(args.hypothesis,args.fit_provenance,args.detector_config,args.capture_dir,
                        args.first_new_utc_day,args.max_mm,args.max_deg)
        with args.output.open('x',encoding='utf-8') as output:
            json.dump(result,output,indent=2,allow_nan=False)
        print(json.dumps({k:result[k] for k in ('status','distinct_new_poses','max_mm','max_deg','consistency_passed')}))
        return 0 if result['consistency_passed'] else 2
    except (OSError,ValueError,KeyError,TypeError) as exc:
        print(str(exc),file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
