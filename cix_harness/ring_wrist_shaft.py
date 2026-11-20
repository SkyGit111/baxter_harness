"""Fit the visible sides of a wooden peg and triangulate its axis across wrist poses."""
import math

import numpy as np

from .pi05_contract import multiply, rotate
from .ring_peg import RingGeometryError


def _camera(state,cfg):
    m=cfg['eef_T_optical'];eq=state['eef_xyzw'];q=multiply(eq,m['xyzw'])
    rot=np.array([rotate(q,v.tolist()) for v in np.eye(3)]).T
    return np.array(cfg['K']),np.array(cfg['D']),rot,np.array(state['eef_m'])+rotate(eq,m['xyz'])


def _rays(uv,K,D,R):
    import cv2
    xy=cv2.undistortPoints(np.asarray(uv,float).reshape(-1,1,2),K,D).reshape(-1,2)
    return np.column_stack((xy,np.ones(len(xy))))@R.T


def shaft_view(bgr,state,ring,peg,cfg,*,allow_partial=False):
    """Locate the receiving colour marker and two sides of its protruding shaft in this view."""
    import cv2
    K,D,R,t=_camera(state,cfg)
    base=np.array(peg['support_center_base_m']);tip=np.array(peg['tip_base_m'])
    cam=(base-t)@R
    if cam[2]<=0:raise RingGeometryError('Wrist is behind the mount')
    expected=cv2.projectPoints(cam,np.zeros(3),np.zeros(3),K,D)[0].ravel()
    hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV);h,s,v=hsv.transpose(2,0,1)
    y,x=np.indices(h.shape)
    view_radius=K[0,0]*ring['outer_radius_m']/cam[2]
    region=np.hypot(x-expected[0],y-expected[1])<view_radius*2.25
    color=np.zeros(h.shape,np.uint8)
    for low,high in cfg['ring_hsv_ranges'][peg['color']]:
        low=list(low);low[1]=max(95,low[1]);low[2]=max(105,low[2])
        color|=cv2.inRange(hsv,np.uint8(low),np.uint8(high))
    if ring.get('wrist_outer_uv_px'):
        cx,cy=ring['wrist_outer_uv_px'];diam=ring['wrist_outer_diameters_px']
        cv2.ellipse(color,(int(round(cx)),int(round(cy))),tuple(int(round(v/2+4.)) for v in diam),
                    ring.get('wrist_outer_angle_deg',0.),0,360,0,-1)
    color[~region]=0
    marks,_=cv2.findContours(color,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    anchors=[]
    for contour in marks:
        if len(contour)<20 or cv2.contourArea(contour)<.65*view_radius**2:continue
        center,axes,_=cv2.fitEllipse(contour)
        if min(axes)>.8*view_radius and max(axes)<3.0*view_radius and math.dist(center,expected)<view_radius*.95:
            anchors.append(center)
    if len(anchors)!=1:raise RingGeometryError('Receiving marker is missing or ambiguous in wrist view')
    anchor=np.asarray(anchors[0]);search=np.hypot(x-anchor[0],y-anchor[1])<max(46.,view_radius*3.6)
    wood=(h>=10)&(h<=42)&(s>=30)&(s<=139)&(v>=175)&search
    comps,labels,stats,_=cv2.connectedComponentsWithStats(np.uint8(wood))
    lines=[]
    for idx in range(1,comps):
        if stats[idx,-1]<42:continue
        yy,xx=np.nonzero(labels==idx);points=np.column_stack((xx,yy))
        if np.linalg.norm(points-anchor,axis=1).min()>view_radius*.6:continue
        mean=points.mean(axis=0);_,values,axes=np.linalg.svd(points-mean,full_matrices=False)
        if values[0]<1.7*values[1]:continue
        direction=axes[0];side=axes[1]
        q=(points-mean)@direction;p=(points-mean)@side
        low,high=np.percentile(q,[3,97]);extent=high-low
        if extent<12:continue
        bins=np.linspace(low+.12*extent,high-.12*extent,8)
        centers=[];widths=[]
        for a,b in zip(bins,bins[1:]):
            in_bin=(q>=a)&(q<=b)
            if in_bin.sum()<6:continue
            mn,mx=np.percentile(p[in_bin],[2,98]);widths.append(mx-mn+1)
            centers.append(mean+direction*(a+b)/2+side*(mn+mx)/2)
        if len(centers)<6:continue
        physical_width=2*peg['radius_m']*K[0,0]/cam[2]
        if not .53*physical_width<np.median(widths)<2.35*physical_width:continue
        pts=np.array(centers);m=pts.mean(axis=0);_,_,axes=np.linalg.svd(pts-m,full_matrices=False)
        axis=axes[0];distance=abs((pts-m)@axes[1])
        if np.percentile(distance,90)>1.10 or abs((anchor-m)@axes[1])>view_radius*.38:continue
        line=m+np.outer([-extent*.4,extent*.4],axis)
        directions=_rays(line,K,D,R);normal=np.cross(*directions);normal/=np.linalg.norm(normal)
        b_ray=_rays([anchor],K,D,R)[0];anchor3=t+b_ray*((base[2]-t[2])/b_ray[2])
        lines.append(dict(camera_origin=t.tolist(),plane_normal=normal.tolist(),support_uv=anchor.tolist(),
                          support_base_m=anchor3.tolist(),line_uv=line.tolist(),fit_p90_px=float(np.percentile(distance,90)),
                          widths_px=widths,line_length_px=extent,edge_pixels=int(len(points)),peg_id=peg['id']))
    if not lines:
        # A pale shaft can saturate against the white board. Use only scanline
        # chords bounded on BOTH sides by the identified coloured support.
        pale=(v>=190)&(s<=108)
        radius=math.ceil(view_radius*1.25);ax,ay=np.rint(anchor).astype(int)
        good=[]
        for swap in (False,True):
            light=pale.T if swap else pale;edges=color.T if swap else color
            cx,cy=(ay,ax) if swap else (ax,ay)
            pts=[];widths=[]
            for row in range(max(0,cy-radius),min(light.shape[0],cy+radius+1)):
                cols=np.flatnonzero(light[row,max(0,cx-radius):min(light.shape[1],cx+radius+1)])+max(0,cx-radius)
                runs=np.split(cols,np.flatnonzero(np.diff(cols)>1)+1)
                candidates=[]
                for run in runs:
                    if not .53*2*peg['radius_m']*K[0,0]/cam[2]<=len(run)<=2.6*2*peg['radius_m']*K[0,0]/cam[2]:continue
                    left,right=int(run[0]),int(run[-1])
                    if not np.any(edges[row,max(0,left-4):left]) or not np.any(edges[row,right+1:right+5]):continue
                    mid=(left+right)/2
                    if math.hypot(mid-cx,row-cy)<1.25*view_radius:candidates.append((mid,len(run)))
                if len(candidates)==1:
                    mid,width=candidates[0];pts.append((row,mid) if swap else (mid,row));widths.append(width)
            if len(pts)<9:continue
            pts=np.asarray(pts);m=pts.mean(axis=0);_,_,axes=np.linalg.svd(pts-m,full_matrices=False)
            axis=axes[0];along=(pts-m)@axis;dist=abs((pts-m)@axes[1]);length=np.ptp(along)
            if length<9 or abs(axis[int(swap)])>.65 or np.percentile(dist,90)>1.:continue
            line=m+np.outer([along.min(),along.max()],axis)
            dirs=_rays(line,K,D,R);normal=np.cross(*dirs);normal/=np.linalg.norm(normal)
            b_ray=_rays([anchor],K,D,R)[0];anchor3=t+b_ray*((base[2]-t[2])/b_ray[2])
            good.append(dict(camera_origin=t.tolist(),plane_normal=normal.tolist(),support_uv=anchor.tolist(),
                support_base_m=anchor3.tolist(),line_uv=line.tolist(),fit_p90_px=float(np.percentile(dist,90)),
                widths_px=widths,line_length_px=float(length),edge_pixels=2*len(pts),peg_id=peg['id'],
                method='shaft_two_edges_against_colour_marker'))
        lines=good
    if len(lines)!=1 and allow_partial:
        if lines:raise RingGeometryError('Multiple shaft candidates in wrist view')
        close=np.hypot(x-anchor[0],y-anchor[1])<view_radius*.95
        px=np.argwhere(wood&close)[:,::-1]
        if len(px)<12:raise RingGeometryError('No current shaft pixels on the receiving colour marker')
        b_ray=_rays([anchor],K,D,R)[0]
        anchor3=t+b_ray*((base[2]-t[2])/b_ray[2])
        return dict(camera_origin=t.tolist(),support_uv=anchor.tolist(),support_base_m=anchor3.tolist(),
                    shaft_pixels=px.tolist(),peg_id=peg['id'],method='current_partial_shaft')
    if len(lines)!=1:raise RingGeometryError('Shaft sides are missing or ambiguous in wrist view')
    return lines[0]


def visible_cap(bgr,state,view,peg,cfg):
    """Use the exposed distal cap arc against a darker background in a side view."""
    import cv2
    K,D,R,t=_camera(state,cfg)
    uv=np.asarray(view['line_uv']);direction=uv[1]-uv[0];direction/=np.linalg.norm(direction)
    anchor=np.asarray(view['support_uv'])
    if (uv.mean(axis=0)-anchor)@direction<0:direction=-direction
    transverse=np.array([-direction[1],direction[0]])
    radius=peg['radius_m']*K[0,0]/((np.asarray(peg['tip_base_m'])-t)@R)[2]
    theta=np.deg2rad(np.linspace(-78,78,17))
    outward=np.cos(theta)[:,None]*direction+np.sin(theta)[:,None]*transverse
    h,w=bgr.shape[:2];bright=bgr.min(axis=2);dark=bgr.max(axis=2)
    best=None
    for length in np.arange(radius*1.4,radius*9,.65):
        for shift in np.arange(-radius,radius+.1,.65):
            origin=anchor+length*direction+shift*transverse
            for rad in np.linspace(radius*.8,radius*1.2,5):
                inside=np.rint(origin+outward*(rad-1.7)).astype(int)
                outside=np.rint(origin+outward*(rad+1.7)).astype(int)
                xy=np.vstack([inside,outside])
                if np.any(xy.min(axis=0)<0) or np.any(xy.max(axis=0)>=[w,h]):continue
                inn=bright[inside[:,1],inside[:,0]].astype(float);out=dark[outside[:,1],outside[:,0]].astype(float)
                good=(inn>155)&(out<140)&(inn-out>55)
                if good.sum()<13 or not good[:4].any() or not good[-4:].any():continue
                score=good.sum()+np.mean(np.clip(inn-out,0,230))/230.
                if best is None or score>best[0]:best=(score,origin,rad,good.sum())
    if best is None:raise RingGeometryError('No distinct distal cap arc against background')
    _,origin,rad,count=best
    # Locate the actual intensity transitions rather than keeping the centre of
    # the first discrete candidate with a high contrast score.
    distances=np.arange(rad-3.3,rad+3.31,.12)
    xy=origin+outward[:,None,:]*distances[None,:,None]
    intensity=cv2.remap(cv2.cvtColor(bgr,cv2.COLOR_BGR2GRAY).astype(np.float32),
                        xy[:,:,0].astype(np.float32),xy[:,:,1].astype(np.float32),cv2.INTER_LINEAR)
    amplitude=np.mean(intensity[:,:4],axis=1)-np.mean(intensity[:,-4:],axis=1)
    valid=amplitude>65.
    if valid.sum()<12:raise RingGeometryError('Insufficient observed cap-to-background transitions')
    half=(np.mean(intensity[:,:4],axis=1)+np.mean(intensity[:,-4:],axis=1))/2
    crossing=np.argmin(abs(intensity-half[:,None]),axis=1)
    points=origin+outward[valid]*distances[crossing[valid],None]
    mean=points.mean(axis=0);relative=points-mean
    param,_,rank,_=np.linalg.lstsq(np.column_stack((2*relative,np.ones(len(points)))),np.sum(relative**2,axis=1),rcond=None)
    if rank!=3:raise RingGeometryError('Degenerate distal cap arc')
    fitted=mean+param[:2];observed_radius=np.median(np.linalg.norm(points-fitted,axis=1))
    # The white end face blooms into a dark background. Report the apparent optical
    # edge separately from the metric shaft radius, and propagate the spread in
    # centres when an intensity transition and the size-constrained arc disagree.
    glow=float(np.median(intensity[valid,:4]))>225.
    if not .75*radius<observed_radius<(1.65 if glow else 1.30)*radius or np.linalg.norm(fitted-origin)>3.5:
        raise RingGeometryError('Distal cap edge fit disagrees with its expected diameter (measured %.2f vs %.2f px, centre change %.2f px)' % (observed_radius,radius,np.linalg.norm(fitted-origin)))
    separation=float(np.linalg.norm(fitted-origin));origin=(origin+fitted)/2;rad=observed_radius
    n=_rays([origin],K,D,R)[0];span=np.array(peg['tip_position_interval_base_m'])[:,2]
    tips=[t+n*((z-t[2])/n[2]) for z in span]
    return dict(wrist_tip_base_m=np.mean(tips,axis=0).tolist(),wrist_tip_uv_px=origin.tolist(),
                tip_interval_base_m=[v.tolist() for v in tips],radius_px=float(rad),arc_samples=int(count),
                optical_center_uncertainty_px=.5+separation/2,transition_center_px=fitted.tolist(),
                edge_p90_px=float(np.percentile(abs(np.linalg.norm(points-fitted,axis=1)-rad),90)),
                cap_ray_base=_rays([origin],K,D,R)[0].tolist(),camera_origin=t.tolist(),
                survey_support_base_m=view['support_base_m'],peg_id=peg['id'])


def triangulate(first,second,peg):
    if first['peg_id']!=second['peg_id'] or first['peg_id']!=peg['id']:raise RingGeometryError('Different peg observations')
    origins=np.asarray([first['camera_origin'],second['camera_origin']])
    baseline=float(np.linalg.norm(origins[1]-origins[0]))
    if baseline<.018:raise RingGeometryError('A distinct wrist position is required to resolve the shaft axis')
    normals=np.array([first['plane_normal'],second['plane_normal']]);matrix=normals[:,:2]
    condition=float(np.linalg.cond(matrix))
    if not np.isfinite(condition) or condition>7.:raise RingGeometryError('Shaft views have too little angular separation')
    heights=np.asarray(peg['tip_position_interval_base_m'])[:,2]
    rhs=(normals*origins).sum(axis=1)
    def intersect(z):
        return np.r_[np.linalg.solve(matrix,rhs-normals[:,2]*z),z]
    top=intersect(float(np.mean(heights)));bottom=intersect(peg['support_z_m'])
    discrepancy=max(math.dist(bottom[:2],v['support_base_m'][:2]) for v in (first,second))
    axis=top-bottom;axis/=np.linalg.norm(axis)
    if axis[2]<.95 or discrepancy>.009:raise RingGeometryError('Two-view shaft disagrees with the receiving marker or upright mount')
    interval=np.array([intersect(v) for v in heights])
    return {'wrist_tip_base_m':top.tolist(),'axis_base':axis.tolist(),'interval_base_m':interval.tolist(),
            'support_base_m':bottom.tolist(),'base_anchor_disagreement_m':discrepancy,
            'view_baseline_m':baseline,'condition':condition,'shaft_views':[first,second]}


def stereo_cap(caps):
    """Intersect independently observed end-cap rays. Z is estimated from parallax."""
    if len(caps)<2 or len({v['peg_id'] for v in caps})!=1:
        raise RingGeometryError('Two independent views of the same peg cap are required')
    origins=np.asarray([v['camera_origin'] for v in caps])
    dirs=np.asarray([v['cap_ray_base'] for v in caps]);dirs/=np.linalg.norm(dirs,axis=1)[:,None]
    bk=max(np.linalg.norm(a-b) for a in origins for b in origins)
    sine=max(np.linalg.norm(np.cross(a,b)) for a in dirs for b in dirs)
    if bk<.018 or sine<.065:raise RingGeometryError('Cap views have insufficient parallax')
    projs=np.eye(3)-dirs[:,:,None]*dirs[:,None,:]
    point=np.linalg.solve(projs.sum(axis=0),np.einsum('nij,nj->i',projs,origins))
    err=np.linalg.norm(np.einsum('nij,nj->ni',projs,point-origins),axis=1)
    base=np.median([v['survey_support_base_m'] for v in caps],axis=0)
    axis=point-base;measured_height=axis[2];axis/=np.linalg.norm(axis)
    # A mismatch with the independent ruler measurement indicates a wrong feature.
    if err.max()>.003 or not .080<measured_height<.110 or axis[2]<.95:
        raise RingGeometryError('Cap rays disagree with the measured mount or protrusion')
    return dict(peg_id=caps[0]['peg_id'],tip_base_m=point.tolist(),support_base_m=base.tolist(),
                axis_base=axis.tolist(),baseline_m=float(bk),ray_sine=float(sine),
                ray_residual_m=err.tolist(),observed_protrusion_m=float(measured_height),
                observations=caps)


def registered_cap(memory,bgr):
    """Allow a fixed cap position only when two distinct current stand features register."""
    import cv2
    anchors=memory.get('mount_anchors',[]);matches=[]
    if memory.get('frame')!='base':return None
    views=memory.get('observations',[])
    if memory.get('method')=='visible_cap_with_measured_protrusion':
        if len(views)!=1 or views[0].get('arc_samples',0)<13:return None
        zrange=np.asarray(views[0].get('tip_interval_base_m',[]))
        if zrange.shape!=(2,3) or not 0.<zrange[1,2]-zrange[0,2]<=.013:return None
    elif memory.get('method')!='stereo_visible_caps' or len(views)<2:return None
    for landmark in anchors:
        u,v=landmark['uv_px'];patch=np.uint8(landmark['bgr']);ph,pw=patch.shape[:2]
        if min(ph,pw)<9 or max(ph,pw)>43 or np.max(np.std(patch.reshape(-1,3),axis=0))<8:continue
        x=u-pw//2-3;y=v-ph//2-3;crop=bgr[y:y+ph+6,x:x+pw+6]
        if crop.shape[:2]!=(ph+6,pw+6):continue
        _,score,_,loc=cv2.minMaxLoc(cv2.matchTemplate(crop,patch,cv2.TM_CCOEFF_NORMED))
        delta=np.array(loc)-3
        if score>.91 and np.linalg.norm(delta)<=1.5:
            matches.append(dict(id=landmark['id'],uv_px=[u,v],delta_px=delta.tolist(),ncc=score))
    if len(matches)<2:return None
    if max(math.dist(a['uv_px'],b['uv_px']) for a in matches for b in matches)<30:return None
    if max(math.dist(a['delta_px'],b['delta_px']) for a in matches for b in matches)>1.5:return None
    return matches
