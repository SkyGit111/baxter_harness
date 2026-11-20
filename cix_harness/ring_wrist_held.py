"""Identify a held ring from the selected wrist's fresh, calibrated close-up."""
import math

import numpy as np

from .pi05_contract import multiply, rotate


def _circle_poses(edge, K, D, radius):
    """Both circular sections of the cone back-projected from the observed ellipse."""
    import cv2
    xy=cv2.undistortPoints(edge.reshape(-1,1,2).astype(float),K,D).reshape(-1,2)
    (x,y),(w,h),angle=cv2.fitEllipse(xy.astype('float32'))
    theta=math.radians(angle)
    rot=np.array([[math.cos(theta),-math.sin(theta)],[math.sin(theta),math.cos(theta)]])
    a=rot@np.diag([4/w**2,4/h**2])@rot.T
    c=np.array([x,y]);q=np.block([[a,(-a@c)[:,None]],[(-c@a)[None,:],np.array([[c@a@c-1]])]])
    vals,axes=np.linalg.eigh(q);l3,l2,l1=vals
    if not l3<0<l2<=l1:return []
    normals=[axes[:,2]*np.sqrt((l1-l2)/(l1-l3))+sign*axes[:,0]*np.sqrt((l2-l3)/(l1-l3)) for sign in (-1,1)]
    rays=np.column_stack((xy,np.ones(len(xy))));poses=[]
    for n in normals:
        if n[2]<0:n=-n
        denom=rays@n
        if denom.min()<.25:continue
        centre=np.linalg.solve(q,n);centre/=centre@n
        distances=np.linalg.norm(rays/denom[:,None]-centre,axis=1)
        scale=radius/np.median(distances)
        poses.append((centre*scale,n,float(np.percentile(abs(distances*scale-radius),90))))
    return poses


def held_ring(bgr,state,profile,scene,prior=None):
    """Require a colour/shape match at the fingers and independent mechanical grip."""
    import cv2
    if state.get('gripping') is not True:return None
    settings=profile['ring_peg'];cfg=settings.get('wrist_alignment',{})
    ident=settings['source_ring_id'];spec=next((x for x in settings['perception']['rings'] if x['id']==ident),None)
    if not spec or not spec.get('loose_candidate') or not cfg:return None
    if any(r['id']==ident and r['location'] in ('loose','on_peg') for r in scene['rings']):return None
    source=cfg.get('surveyed_ring_sizes',{}).get(ident)
    if source:
        inner,outer=source['inner_radius_m'],source['outer_radius_m']
        size_uncertainty=source['outer_radius_uncertainty_m']
    elif prior and prior.get('skill')=='pick' and prior.get('ring_id')==ident:
        inner,outer=prior['inner_radius_m'],prior['outer_radius_m']
        size_uncertainty=settings['horizontal_uncertainty_m']
    else:return None
    K=np.asarray(cfg['K']);D=np.asarray(cfg['D']);mount=cfg['eef_T_optical'];eq=state['eef_xyzw']
    q=multiply(eq,mount['xyzw']);R=np.array([rotate(q,v.tolist()) for v in np.eye(3)]).T
    t=np.array(state['eef_m'])+rotate(eq,mount['xyz'])
    tcp=np.array(state['eef_m'])+rotate(eq,profile['tcp_offset_tool_m'])
    tcp_cam=(tcp-t)@R
    if tcp_cam[2]<=0 or list(bgr.shape[1::-1])!=cfg['resolution']:return None
    uv=cv2.projectPoints(tcp_cam,np.zeros(3),np.zeros(3),K,D)[0].ravel()
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV);mask=np.zeros(hsv.shape[:2],np.uint8)
    for low,hi in cfg['ring_hsv_ranges'][spec['color']]:
        low=list(low);low[1]=max(low[1],95);low[2]=max(low[2],105)
        mask|=cv2.inRange(hsv,np.uint8(low),np.uint8(hi))
    contours,tree=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    if tree is None:return None
    objects=[]
    for i,(edge,links) in enumerate(zip(contours,tree[0])):
        if links[3]>=0 or links[2]<0 or len(edge)<35:continue
        (x,y),axes,_=cv2.fitEllipse(edge)
        if min(axes)<18 or min(axes)/max(axes)<.60 or math.dist([x,y],uv)>max(axes)*.58:continue
        for hole,branch in zip(contours,tree[0]):
            if branch[3]!=i or len(hole)<16:continue
            (hx,hy),holes,_=cv2.fitEllipse(hole)
            if min(holes)<5:continue
            # The same-colour mount can merge into part of the held ring's outline.
            # Only measured outer arcs surrounding the independently observed bore vote.
            points=edge.reshape(-1,2);offsets=points-[hx,hy]
            angles=(np.arctan2(offsets[:,1],offsets[:,0])+np.pi)*24/(2*np.pi)
            bins=angles.astype(int)%24;radii=np.linalg.norm(offsets,axis=1)
            sectors=[np.median(radii[bins==j]) for j in range(24) if np.any(bins==j)]
            if len(sectors)<19:continue
            mid=np.median(sectors);keep=(radii>.78*mid)&(radii<1.16*mid)
            if len(np.unique(bins[keep]))<17:continue
            arc=points[keep]
            (x,y),axes,ellipse_angle=cv2.fitEllipse(arc)
            if abs(np.mean(holes)/np.mean(axes)-inner/outer)>.09 or math.dist([x,y],[hx,hy])>.12*max(axes):continue
            options=_circle_poses(arc,K,D,outer)
            possible=[]
            for camera_center,n,error in options:
                centre=t+R@camera_center;normal=R@n
                if normal[2]<0:normal=-normal
                if error>.0025 or np.linalg.norm(centre[:2]-tcp[:2])>outer+.008 or abs(centre[2]-tcp[2])>.042:continue
                possible.append((centre,normal,error,camera_center[2]))
            if not possible:continue
            center=np.mean([p[0] for p in possible],axis=0)
            plane=profile['table_plane']
            if np.dot(center,plane['normal'])+plane['offset']<.045:continue
            # For release height/slope calculations use the steepest credible face.
            # A single circular silhouette often admits two distinct face normals.
            worst=min(possible,key=lambda p:p[1][2]);normal=worst[1]
            depth_scale_uncertainty=float(abs(size_uncertainty/outer*worst[3]))
            objects.append({'id':ident,'kind':'ring','color':spec['color'],'location':'held','peg_id':None,
                'center_base_m':center.tolist(),'normal_base':normal.tolist(),'inner_radius_m':inner,'outer_radius_m':outer,
                'inner_boundary_p90_m':size_uncertainty,'outer_boundary_p90_m':worst[2],
                'depth_plane_p90_m':depth_scale_uncertainty,'measurement_source':'selected_wrist_calibrated_annulus',
                'wrist_hole_uv_px':[hx,hy],'wrist_outer_uv_px':[x,y],'wrist_outer_diameters_px':list(axes),
                'wrist_outer_angle_deg':ellipse_angle,
                'wrist_hole_diameters_px':list(holes),'wrist_center_to_tcp_m':(center-tcp).tolist(),
                'source_ring_visible_on_table':False,
                'wrist_size_depth_uncertainty_m':depth_scale_uncertainty,'candidate_normals_base':[p[1].tolist() for p in possible]})
    return objects[0] if len(objects)==1 else None
