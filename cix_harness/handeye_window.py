"""Estimate one synchronized wrist/marker pose per stationary window, using only that window."""
import math

import numpy as np

from .handeye import angle, closest_rotation, transform
from .handeye_capture import window_status


def estimate_rows(data,method):
    if method=='last-frame':return data,None
    if method!='stationary-window' or not isinstance(data,list):
        raise ValueError('Stationary-window estimation requires captured per-pose records')
    rows=[];diagnostics=[]
    for sample in data:
        frames=sample.get('stationary_frames')
        if not isinstance(frames,list) or not frames:
            raise ValueError('Missing synchronized stationary frames for '+str(sample.get('drive_pose_id')))
        error=window_status(frames)
        if error:raise ValueError(error)
        pose_id=sample.get('drive_pose_id')
        if (not pose_id or any(f.get('drive_pose_id')!=pose_id for f in frames)
                or ('color_frame' in sample and sample['color_frame']!=frames[-1].get('color_frame'))):
            raise ValueError('Stationary frames belong to a different pose or endpoint frame')
        corners=np.asarray([f['corners_optical_m'] for f in frames],dtype=float)
        uv=np.asarray([f['corners_px'] for f in frames],dtype=float)
        if (corners.shape!=(len(frames),4,3) or uv.shape!=(len(frames),4,2) or
                not np.isfinite(corners).all() or not np.isfinite(uv).all() or np.min(corners[:,:,2])<=0):
            raise ValueError('Invalid decoded RGB-D corners in stationary window')
        sizes=sample['ruler_reference']
        x,y=sizes['width_m']/2,sizes['height_m']/2
        if not np.isfinite([x,y]).all() or min(x,y)<=0:raise ValueError('Measured physical marker dimensions required')
        model=np.array([[-x,-y,0],[x,-y,0],[x,y,0],[-x,y,0]])
        med=np.median(corners,axis=0);center=med.mean(axis=0)
        cov=(med-center).T@model
        if np.linalg.matrix_rank(cov)<2:raise ValueError('Degenerate stationary marker corner geometry')
        camera=np.eye(4);camera[:3,:3]=closest_rotation(cov);camera[:3,3]=center
        arms=np.asarray([transform(f['base_T_effector']) for f in frames])
        wrist=np.eye(4);wrist[:3,3]=np.median(arms[:,:3,3],axis=0)
        wrist[:3,:3]=closest_rotation(arms[:,:3,:3].mean(axis=0))
        tags=np.asarray([transform(f['camera_T_tag']) for f in frames])
        normal_deg=np.degrees(np.arccos(np.clip(tags[:,:3,2]@camera[:3,2],-1,1)))
        center_mm=1000*np.linalg.norm(tags[:,:3,3]-center,axis=1)
        diagnostics.append({'id':pose_id,'frame_count':len(frames),'span_s':frames[-1]['host_time_s']-frames[0]['host_time_s'],
                            'center_rms_mm':float(np.sqrt(np.mean(center_mm**2))),'center_max_mm':float(np.max(center_mm)),
                            'normal_rms_deg':float(np.sqrt(np.mean(normal_deg**2))),'normal_max_deg':float(max(normal_deg)),
                            'last_center_difference_mm':float(center_mm[-1]),'last_normal_difference_deg':float(normal_deg[-1]),
                            'last_rotation_difference_deg':math.degrees(angle(tags[-1,:3,:3].T@camera[:3,:3]))})
        rows.append({**sample,'base_T_effector':wrist.tolist(),'camera_T_tag':camera.tolist(),
                     'corners_px':np.median(uv,axis=0).tolist(),'corners_optical_m':med.tolist()})
    return rows,diagnostics
