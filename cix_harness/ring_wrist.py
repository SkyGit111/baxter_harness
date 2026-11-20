"""Measure the hole-to-tip displacement in the selected wrist camera's current image."""
import math

import numpy as np

from .contract import vector
from .pi05_contract import multiply,rotate
from .ring_peg import RingGeometryError, RingViewConflict


def _mask_roi(hsv,uv,radius,ranges):
    import cv2
    h,w=hsv.shape[:2];x,y=np.rint(uv).astype(int);r=int(math.ceil(radius))
    if not (x-r>=0 and x+r<w and y-r>=0 and y+r<h):raise RingGeometryError('Wrist landmark is outside the calibrated image')
    crop=hsv[y-r:y+r+1,x-r:x+r+1];mask=np.zeros(crop.shape[:2],np.uint8)
    for low,high in ranges:
        mask|=cv2.inRange(crop,np.array(low,np.uint8),np.array(high,np.uint8))
    contours,tree=cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE,offset=(x-r,y-r))
    return contours,tree


def _ellipse(contours,tree,reference,expected_radius,hole):
    import cv2
    matches=[]
    if tree is None:raise RingGeometryError('No wrist landmark contour')
    for contour,links in zip(contours,tree[0]):
        if (links[3]>=0)!=hole or len(contour)<12:continue
        pts=contour.reshape(-1,2);(x,y),axes,angle=cv2.fitEllipse(contour);axes=np.asarray(axes)*.5
        if axes.min()<2 or not .56*expected_radius<np.mean(axes)<1.55*expected_radius:continue
        if axes.min()/axes.max()<.53 or math.dist([x,y],reference)>max(6,expected_radius*1.6):continue
        a=math.radians(angle);rot=np.array([[math.cos(a),-math.sin(a)],[math.sin(a),math.cos(a)]])
        radial=np.linalg.norm(((pts-[x,y])@rot)/axes,axis=1)
        residual=float(np.percentile(abs(radial-1),90)*np.mean(axes))
        if residual>1.5:continue
        matches.append(([x,y],max(.5,residual),pts))
    if len(matches)!=1:raise RingGeometryError('Missing or ambiguous wrist {} contour'.format('hole' if hole else 'peg tip'))
    return matches[0]


def _metric_rim(edge,K,D,R,t,landmark,normal):
    import cv2
    rays=cv2.undistortPoints(edge.astype(float).reshape(-1,1,2),K,D).reshape(-1,2)
    dirs=np.column_stack((rays,np.ones(len(rays))))@R.T
    n=np.asarray(normal,float);n=n/np.linalg.norm(n);denom=dirs@n
    if min(abs(denom))<.3:raise RingGeometryError('Wrist edge nearly parallel to measured face')
    scale=np.dot(landmark-t,n)/denom
    if np.any(scale<=0):raise RingGeometryError('Wrist edge intersects behind camera')
    world=t+dirs*scale[:,None];ref=world.mean(axis=0)
    basis=np.cross(n,[1.,0.,0.]);basis/=np.linalg.norm(basis);xy=(world-ref)@np.stack((basis,np.cross(n,basis)),axis=1)
    mid,_,rank,_=np.linalg.lstsq(np.column_stack((2*xy,np.ones(len(xy)))),np.sum(xy**2,axis=1),rcond=None)
    if rank!=3:raise RingGeometryError('Degenerate wrist metric contour')
    radii=np.linalg.norm(xy-mid[:2],axis=1);radius=float(np.median(radii))
    err=float(np.percentile(abs(radii-radius),90))+.5*max(scale)/min(K[0,0],K[1,1])/min(abs(denom))
    return radius,err


def measure(bgr,state,ring,peg,config):
    """Use commissioned optical extrinsics and separate D455 heights to account for parallax."""
    import cv2
    h,w=bgr.shape[:2];K=np.asarray(config['K'],dtype=float);D=np.asarray(config['D'],dtype=float)
    if [w,h]!=config['resolution'] or K.shape!=(3,3) or min(K[0,0],K[1,1])<=0 or not np.isfinite(K).all() or not np.isfinite(D).all():
        raise RingGeometryError('Wrist CameraInfo does not match the acquired image')
    mounting=config['eef_T_optical'];eq=state['eef_xyzw'];q=multiply(eq,mounting['xyzw'])
    R=np.array([rotate(q,a.tolist()) for a in np.eye(3)]).T
    t=np.asarray(state['eef_m'])+rotate(eq,vector(mounting['xyz'],3,'wrist optical translation'))
    landmarks=np.asarray([ring['center_base_m'],peg['tip_base_m']],float);cam=(landmarks-t)@R
    if not np.all(np.isfinite(cam)) or np.any(cam[:,2]<=0):raise RingGeometryError('Wrist registration projects behind image')
    projected=cv2.projectPoints(cam,np.zeros(3),np.zeros(3),K,D)[0].reshape(-1,2)
    expected=K[0,0]*np.array([ring['inner_radius_m'],peg['radius_m']])/cam[:,2];hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV)
    if min(expected)<2.5:raise RingGeometryError('Wrist view does not resolve the ring hole and peg tip')
    ring_roi=K[0,0]*ring['outer_radius_m']/cam[0,2]+config.get('search_margin_px',12)
    rc,rt=_mask_roi(hsv,projected[0],ring_roi,config['ring_hsv_ranges'][ring['color']]);pc,pt=_mask_roi(hsv,projected[1],expected[1]+config.get('search_margin_px',12),config['peg_tip_hsv_ranges'])
    hole,p_error,rim_points=_ellipse(rc,rt,projected[0],expected[0],True)
    tip,q_error,tip_points=_ellipse(pc,pt,projected[1],expected[1],False)
    xy=cv2.undistortPoints(np.array([hole,tip],np.float64).reshape(-1,1,2),K,D).reshape(-1,2)
    directions=np.column_stack((xy,np.ones(2)))@R.T
    if min(abs(directions[:,2]))<.3:raise RingGeometryError('Wrist rays do not resolve tabletop offsets')
    scale=(landmarks[:,2]-t[2])/directions[:,2]
    if np.any(scale<=0):raise RingGeometryError('Wrist heights intersect behind camera')
    points=t+directions*scale[:,None];offset=points[1,:2]-points[0,:2]
    mm_per_px=scale/np.min([K[0,0],K[1,1]])/abs(directions[:,2]);slope=np.linalg.norm(directions[:,:2],axis=1)/abs(directions[:,2])
    uncertainty=float(np.dot(mm_per_px,[p_error,q_error])+slope.sum()*config['height_uncertainty_m']+config['relative_xy_uncertainty_m'])
    disagreement=float(np.linalg.norm((landmarks[1,:2]-landmarks[0,:2])-offset))
    if disagreement>config['cross_camera_tolerance_m']:raise RingViewConflict('D455 and wrist disagree on hole-to-tip displacement')
    inner,inner_error=_metric_rim(rim_points,K,D,R,t,landmarks[0],ring['normal_base'])
    radius,rod_error=_metric_rim(tip_points,K,D,R,t,landmarks[1],peg['axis_base'])
    return {'offset_xy_m':offset.tolist(),'uncertainty_m':uncertainty,'hole_uv_px':hole,'peg_uv_px':tip,
            'inner_radius_m':inner,'inner_radius_error_m':inner_error,'peg_radius_m':radius,'peg_radius_error_m':rod_error,
            'hole_edge_px':len(rim_points),'peg_edge_px':len(tip_points),'inter_camera_disagreement_m':disagreement}
