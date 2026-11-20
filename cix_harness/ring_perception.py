"""Observe coloured annular faces and post tips in a single aligned D455 RGB-D frame."""
import math

import numpy as np

from .pi05_contract import rotate


def _circle(points):
    xy=points[:,:2]
    centre=xy.mean(axis=0)
    local=xy-centre
    a=np.column_stack((2*local,np.ones(len(xy))))
    p,_,rank,_=np.linalg.lstsq(a,np.sum(local**2,axis=1),rcond=None)
    if rank!=3:raise ValueError('Degenerate circular boundary')
    ctr=centre+p[:2]
    distances=np.linalg.norm(xy-ctr,axis=1)
    radius=float(np.median(distances))
    return ctr,radius,float(np.percentile(abs(distances-radius),90))


def _roi(spec,width,height):
    x0,y0,x1,y1=np.rint(spec['roi_px']).astype(int)
    if not 0<=x0<x1<width or not 0<=y0<y1<height:
        raise ValueError('Object ROI must lie within the actual D455 frame')
    return int(x0),int(y0),int(x1),int(y1)


class _Geometry:
    def __init__(self,depth,intrinsics,profile):
        self.depth=depth;self.intrinsics=intrinsics
        q=profile['camera_to_base_xyzw']
        self.R=np.asarray([rotate(q,v.tolist()) for v in np.eye(3)]).T
        self.t=np.asarray(profile['camera_to_base_translation_m'])
        plane=profile['table_plane'];self.n=np.asarray(plane['normal'],dtype=float);length=np.linalg.norm(self.n)
        if not np.isfinite(length) or length<.9:raise ValueError('Missing measured table normal')
        self.n=self.n/length;self.d=float(plane['offset'])/length
        tangent=np.array([1.,0,0])-self.n*self.n[0]
        self.ex=tangent/np.linalg.norm(tangent);self.ey=np.cross(self.n,self.ex)
        self.basis=np.asarray([self.ex,self.ey,self.n]).T

    def rays(self,pixels):
        points=np.asarray(pixels,dtype=float).reshape(-1,2)
        if not len(points):return np.empty((0,3),dtype=float)
        if callable(getattr(self.intrinsics,'rays',None)):
            rays=self.intrinsics.rays(points)
        else:
            import pyrealsense2 as rs
            rays=[rs.rs2_deproject_pixel_to_point(self.intrinsics,p.tolist(),1.) for p in points]
        return np.asarray(rays)@self.R.T

    def cloud(self,pixels):
        pixels=np.asarray(pixels,dtype=int).reshape(-1,2)
        z=self.depth[pixels[:,1],pixels[:,0]]
        good=np.isfinite(z)&(z>.1)&(z<3.)
        return self.t+self.rays(pixels[good])*z[good,None]

    def project(self,point):
        camera=(np.asarray(point)-self.t)@self.R
        if camera[2]<=0:return None
        if callable(getattr(self.intrinsics,'project',None)):
            return self.intrinsics.project(camera)
        import pyrealsense2 as rs
        return rs.rs2_project_point_to_pixel(self.intrinsics,camera.tolist())

    def on_plane(self,pixels,normal,offset):
        rays=self.rays(pixels)
        denom=rays@normal
        if np.any(abs(denom)<.1):raise ValueError('Unstable intersection with object top plane')
        z=-(self.t@normal+offset)/denom
        if np.any(z<=0):raise ValueError('Estimated surface behind camera')
        return self.t+rays*z[:,None]


def _face_geometry(mask,outer,inner,geometry):
    import cv2
    region=np.zeros_like(mask)
    cv2.drawContours(region,[outer],0,255,-1)
    if inner is not None:cv2.drawContours(region,[inner],0,0,-1)
    region=cv2.bitwise_and(region,mask)
    interior=cv2.erode(region,np.ones((3,3),np.uint8))
    v,u=np.nonzero(interior)
    if len(u)<12:raise ValueError('Face too small to isolate from its boundary')
    ids=np.linspace(0,len(u)-1,min(240,len(u))).astype(int)
    cloud=geometry.cloud(np.column_stack((u[ids],v[ids])))
    if len(cloud)<max(10,.58*len(ids)):raise ValueError('Missing aligned face depth')
    # A candidate plane must explain a majority of independent interior depth pixels.
    heights=cloud@geometry.n+geometry.d
    use=abs(heights-np.median(heights))<=.009
    if use.sum()<max(10,.65*len(ids)):raise ValueError('Mixed foreground and background depths')
    kept=cloud[use];mid=kept.mean(axis=0)
    _,singular,axes=np.linalg.svd(kept-mid,full_matrices=False)
    normal=axes[-1];normal=normal if normal@geometry.n>=0 else -normal
    residual=(kept-mid)@normal
    depth_p90=float(np.percentile(abs(residual),90))
    spread_m=float(singular[1]/np.sqrt(len(kept)))
    tilt=float(np.degrees(np.arccos(np.clip(normal@geometry.n,-1.,1.))))
    if spread_m<.0015 or depth_p90>.006:raise ValueError('Depth does not resolve a planar face')
    # The face normal of a small peg cap is often dominated by quantization.
    # Keep that uncertainty explicit in the observation.
    fit_normal=normal if spread_m>=.004 else geometry.n
    offset=-float(fit_normal@mid)
    ex=geometry.ex-fit_normal*(fit_normal@geometry.ex);ex=ex/np.linalg.norm(ex)
    basis=np.asarray([ex,np.cross(fit_normal,ex),fit_normal]).T
    def boundary(contour):
        pix=contour.reshape(-1,2)
        world=geometry.on_plane(pix,fit_normal,offset)
        center,radius,error=_circle(world@basis)
        return (np.r_[center,-offset]@basis.T),radius,error
    center,outer_r,outer_err=boundary(outer)
    uv=np.mean(outer.reshape(-1,2),axis=0)
    stat={'center_uv_px':uv.tolist(),'center_base_m':center.tolist(),'top_height_above_table_m':float(center@geometry.n+geometry.d),
          'face_normal_base':normal.tolist(),'normal_base':normal.tolist(),'tilt_deg':tilt,'normal_resolved':spread_m>=.004,
          'depth_inliers':int(len(kept)),'depth_sampled':int(len(ids)),'depth_plane_p90_m':depth_p90,
          'outer_radius_m':outer_r,'outer_boundary_p90_m':outer_err,'inner_radius_m':None}
    if outer_r<.002 or outer_err>max(.003,.18*outer_r):raise ValueError('Outer contour inconsistent with a circular face')
    if inner is not None:
        hole,hole_r,hole_err=boundary(inner)
        if hole_r<.001 or outer_r<=hole_r or hole_err>max(.0025,.2*hole_r) or np.linalg.norm(hole-center)>max(.005,.18*outer_r):
            raise ValueError('Hole inconsistent with a concentric annulus')
        center=(center+hole)/2;stat['center_base_m']=center.tolist()
        stat.update(inner_radius_m=hole_r,inner_boundary_p90_m=hole_err,
                    rim_grasp_candidates_base_m=[(center+(basis[:,0]*math.cos(t)+basis[:,1]*math.sin(t))*(outer_r+hole_r)/2).tolist()
                                                 for t in np.linspace(0,2*np.pi,8,endpoint=False)])
        # UVs come from real contour/ray samples. A selected rim pixel is inside the colour mask.
        xy=np.column_stack((u,v));world=geometry.on_plane(xy,fit_normal,offset)
        rim=stat['rim_grasp_candidates_base_m']
        stat['rim_grasp_candidates_uv_px']=[xy[np.linalg.norm(world-p,axis=1).argmin()].tolist() for p in rim]
        stat['rim_grasp_uv_px']=stat['rim_grasp_candidates_uv_px'][0];stat['rim_grasp_base_m']=rim[0]
    return stat


def _detect(spec,kind,hsv,geometry,allowed=None):
    import cv2
    h,w=geometry.depth.shape;x0,y0,x1,y1=_roi(spec,w,h)
    roi=hsv[y0:y1+1,x0:x1+1]
    local=np.zeros(roi.shape[:2],np.uint8)
    for low,high in spec['hsv_ranges']:
        local|=cv2.inRange(roi,np.asarray(low,np.uint8),np.asarray(high,np.uint8))
    mask=np.zeros((h,w),np.uint8);mask[y0:y1+1,x0:x1+1]=local
    if allowed is not None:mask=cv2.bitwise_and(mask,allowed)
    contours,hierarchy=cv2.findContours(mask,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_NONE)
    results=[]
    if hierarchy is None:return results
    for index,outer in enumerate(contours):
        family=hierarchy[0][index]
        if family[3]!=-1 or len(outer)<8 or cv2.contourArea(outer)<spec.get('min_area_px',14):continue
        px=outer.reshape(-1,2)
        if np.any(px.min(axis=0)<=[x0,y0]) or np.any(px.max(axis=0)>=[x1,y1]):continue
        inner=None
        if kind=='ring':
            holes=[c for j,c in enumerate(contours) if hierarchy[0][j][3]==index and len(c)>=6]
            if not holes:continue
            inner=max(holes,key=cv2.contourArea)
            ratio=cv2.contourArea(inner)/cv2.contourArea(outer)
            if not .025<=ratio<=.82:continue
        try:row=_face_geometry(mask,outer,inner,geometry)
        except ValueError:continue
        lo,hi=spec.get('radius_range_m',[.002,.15])
        if not lo<=row['outer_radius_m']<=hi:continue
        row.update(id=spec['id'],kind=kind,color=spec.get('color'),peg_id=None)
        if kind=='peg':
            resolved=row['normal_resolved'] and row['depth_plane_p90_m']<.0025
            row.update(tip_uv_px=row['center_uv_px'],tip_base_m=row['center_base_m'],radius_m=row['outer_radius_m'],
                       axis_base=row['face_normal_base'] if resolved else None,support_z_m=None)
        results.append(row)
    return results


def _ordered_quad(points):
    p=np.asarray(points,np.float32).reshape(4,2)
    indices=np.argsort(np.arctan2(*(p-p.mean(axis=0)).T[::-1]))
    p=p[indices]
    return np.roll(p,-np.argmin(p.sum(axis=1)),axis=0)


def _registered_peg(spec,board,bgr,geom):
    """Reobserve the rigid board and local rod appearance before using a measured height interval."""
    import cv2
    ref=spec.get('bracket_reference')
    if board is None or not ref:return None
    source=_ordered_quad(ref['board_quad_px']);actual=_ordered_quad(board)
    corner_drift=float(np.max(np.linalg.norm(source-actual,axis=1)))
    if corner_drift>ref.get('max_board_drift_px',2.5):return None
    warp=cv2.getPerspectiveTransform(source,actual)
    transform=lambda p:cv2.perspectiveTransform(np.array([[p]],np.float32),warp)[0,0]
    pred=transform(ref['patch_center_px']);tip=transform(ref['tip_px']);support=transform(ref['support_px'])
    patch=np.asarray(ref['appearance_bgr'],dtype=np.uint8);h,w=patch.shape[:2]
    if patch.ndim!=3 or patch.shape[-1]!=3 or min(w,h)<7 or w%2==0 or h%2==0 or max(w,h)>65:return None
    if float(np.max(patch.reshape(-1,3).std(axis=0)))<5:return None
    x,y=np.rint(pred).astype(int);sx,sy=x-w//2-3,y-h//2-3
    crop=bgr[sy:sy+h+6,sx:sx+w+6]
    if crop.shape[:2]!=(h+6,w+6):return None
    scores=cv2.matchTemplate(crop,patch,cv2.TM_CCOEFF_NORMED)
    _,score,_,origin=cv2.minMaxLoc(scores)
    if not np.isfinite(score) or score<ref.get('min_appearance_ncc',.85):return None
    shift=np.array([sx+origin[0]+w//2,sy+origin[1]+h//2])-pred
    if np.linalg.norm(shift)>2.5:return None
    tip+=shift;support+=shift
    # Directly recheck the present marker pixels around the tip. Correlation of the larger tile
    # alone can remain high when the gripper covers the cap but the surrounding coloured ring stays visible.
    reference_tip=np.rint(np.asarray(ref['tip_px'])-ref['patch_center_px']+[w//2,h//2]).astype(int)
    ux,uy=np.rint(tip).astype(int);rx,ry=reference_tip
    cap_now=bgr[uy-2:uy+3,ux-2:ux+3].astype(float);cap_ref=patch[ry-2:ry+3,rx-2:rx+3].astype(float)
    if cap_ref.shape!=(5,5,3) or cap_now.shape!=cap_ref.shape:return None
    def chroma(a):return a/np.maximum(1.,a.sum(axis=2,keepdims=True))
    cap_chroma_rms=float(np.sqrt(np.mean((chroma(cap_now)-chroma(cap_ref))**2)))
    if cap_chroma_rms>.045:return None
    h0=float(ref['observed_stack_height_m']);limits=np.asarray(ref['protrusion_m'],dtype=float)
    if limits.shape!=(2,) or not 0<limits[0]<=limits[1]:return None
    tip_interval=[geom.on_plane([tip],geom.n,geom.d-h0-limit)[0] for limit in limits]
    pt=np.mean(tip_interval,axis=0);pc=geom.on_plane([support],geom.n,geom.d-h0)[0]
    shaft=pt-pc;shaft/=np.linalg.norm(shaft)
    return {'id':spec['id'],'kind':'peg','color':spec['color'],'tip_uv_px':tip.tolist(),'tip_base_m':pt.tolist(),
            'radius_m':float(ref['radius_m']),'radius_interval_m':ref['radius_interval_m'],'axis_base':shaft.tolist(),
            'support_z_m':float(pc[2]),'support_center_base_m':pc.tolist(),'support_uv_px':support.tolist(),
            'stack_bottom_to_top':None,'observed_ring_ids':[],'tip_position_interval_base_m':[v.tolist() for v in tip_interval],
            'measurement_source':'current_board_and_cap_appearance_registered_to_exposure_bracket',
            'appearance_ncc':score,'current_cap_chroma_rms':cap_chroma_rms,'board_corner_drift_px':corner_drift,
            'depth_support_spread_m':ref['depth_support_spread_m'],'current_tip_depth_measured':False,
            'tip_position_uncertainty_m':ref['tip_position_uncertainty_m'],'observed_rim_color':spec['color']}


def _local_table(ring,bgr,geometry):
    """Use eight visibly bare angular sectors around the annular face as local table references."""
    u,v=ring['center_uv_px'];h,w=geometry.depth.shape
    dx,dy=np.meshgrid(np.arange(-24,25),np.arange(-24,25))
    rad=np.hypot(dx,dy);x=np.rint(u+dx).astype(int);y=np.rint(v+dy).astype(int)
    keep=(rad>=11)&(rad<=21)&(x>=0)&(x<w)&(y>=0)&(y<h)
    x,y,dx,dy=[z[keep] for z in (x,y,dx,dy)]
    dark=bgr[y,x].max(axis=1)<145
    group=np.floor((np.arctan2(dy,dx)+np.pi)*4/np.pi).astype(int)%8
    cx,cy=ring['center_base_m'][:2];n=geometry.n
    sector_medians=[];pieces=[]
    for i in range(8):
        ok=(group==i)&dark
        points=geometry.cloud(np.column_stack((x[ok],y[ok]))[::2])
        if len(points)<12:continue
        heights=points@n+geometry.d
        if abs(np.median(heights))>.018 or np.percentile(heights,90)-np.percentile(heights,10)>.016:continue
        # Correct sample Z for the known table slope at the ring XY.
        z_here=points[:,2]+((points[:,0]-cx)*n[0]+(points[:,1]-cy)*n[1])/n[2]
        sector_medians.append(float(np.median(z_here)));pieces.append(z_here)
    if len(pieces)<6:return
    vals=np.concatenate(pieces);mid=float(np.median(sector_medians))
    upper=max(float(np.percentile(vals,95)),max(sector_medians))
    ring.update(local_table_z_m=mid,local_table_upper_z_m=upper,
                local_table_sector_centres_z_m=sector_medians,local_table_depth_count=len(vals),
                thickness_from_local_table_m=float(ring['center_base_m'][2]-mid))


def _stack_surface(peg,spec,hsv,geom):
    import cv2
    support=peg.get('support_uv_px');tip=peg['tip_uv_px']
    ref=spec.get('bracket_reference',{})
    ranges=ref.get('support_hsv_ranges',[])
    if not support or not ranges:return
    h,w=geom.depth.shape;u,v=np.rint(support).astype(int)
    radius=ref.get('ring_radius_px',6.5)
    y,x=np.mgrid[max(0,v-9):min(h,v+10),max(0,u-9):min(w,u+10)]
    d=np.hypot(x-support[0],y-support[1]);pix=np.stack((x,y),axis=-1)
    a=np.asarray(tip)-support;off=pix-np.asarray(support)
    proj=np.clip((off@a)/max(1e-6,a@a),0.,1.)
    shaft_dist=np.linalg.norm(off-proj[:,:,None]*a,axis=-1)
    area=(d>radius*.4)&(d<radius*.92)&(shaft_dist>1.8)
    colors=np.zeros(x.shape,np.uint8)
    for low,high in ranges:colors|=cv2.inRange(hsv[y,x],np.uint8(low),np.uint8(high))
    measured=pix[area&(colors>0)];pts=geom.cloud(measured);height=pts@geom.n+geom.d
    peg['protected_color_face_visible']=len(measured)>=12
    peg['protected_color_pixels']=len(measured)
    if 'reference_depth_patch_mm' in ref and len(measured):
        old=np.asarray(ref['reference_depth_patch_mm'],float)*.001
        centre=np.asarray(ref['patch_center_px']);shift=np.asarray(tip)-ref['tip_px']
        index=np.rint(measured-shift-centre+[old.shape[1]//2,old.shape[0]//2]).astype(int)
        inside=(index[:,0]>=0)&(index[:,1]>=0)&(index[:,0]<old.shape[1])&(index[:,1]<old.shape[0])
        index=index[inside];match=measured[inside]
        depth_now=geom.depth[match[:,1],match[:,0]];depth_old=old[index[:,1],index[:,0]]
        use=(depth_now>.1)&(depth_now<3.)&(depth_old>.1)&(depth_old<3.)
        match=match[use];rise=(depth_now[use]-depth_old[use])*(geom.rays(match)@geom.n)
        if len(rise)>=12:
            delta=match-support;angles=np.arctan2(delta[:,1],delta[:,0])
            bins=np.floor((angles+np.pi)*4/np.pi).astype(int)%8
            sectors=[float(np.median(rise[bins==i])) for i in range(8) if np.sum(bins==i)>=3]
            if len(sectors)>=3:
                q25,median,q75=np.percentile(sectors,[25,50,75])
                peg.update(stack_depth_delta_m=float(median),stack_delta_sector_medians_m=sectors,
                           stack_delta_paired_pixels=len(rise),stack_delta_from_frame=ref.get('reference_frame'))
                # Report a rise separately from task success. The harness must also verify the release,
                # absence of a loose ring, and consistency with the newly added ring's thickness.
                peg['stack_rise_supported']=bool(q25>.005 and q75-q25<.008 and median>.007)
    if len(pts)<12 or len(pts)<.55*len(measured):return
    p10,median,p90=np.percentile(height,[10,50,90])
    if p90-p10>.016 or not .01<median<.28:return
    base=geom.on_plane([support],geom.n,geom.d-median)[0]
    peg.update(current_support_z_m=float(base[2]),current_support_base_m=base.tolist(),current_support_depth_count=len(pts),
               current_support_interval_height_m=[float(p10),float(p90)],current_support_height_m=float(median),
               current_support_height_change_m=float(median-ref['observed_stack_height_m']))


def observe_ring_scene(bgr,depth_m,intrinsics,profile,settings,state):
    """Use only tiles in this frame. Missing/ambiguous contours remain explicit observations."""
    import cv2
    bgr=np.asarray(bgr);depth=np.asarray(depth_m)
    if depth.ndim!=2 or bgr.shape!=(*depth.shape,3) or bgr.dtype!=np.uint8:raise ValueError('Expected aligned uint8 BGR and metric depth')
    h,w=depth.shape;hsv=cv2.cvtColor(bgr,cv2.COLOR_BGR2HSV);geometry=_Geometry(depth,intrinsics,profile)
    scene={'rings':[],'pegs':[],'missing':[],'held_ring_id':None,'resolution':[w,h],'board_polygon_px':None}
    board=None
    if settings.get('board'):
        x0,y0,x1,y1=_roi(settings['board'],w,h)
        panel=hsv[y0:y1+1,x0:x1+1];light=cv2.inRange(panel,np.array([0,0,155],np.uint8),np.array([179,100,255],np.uint8))
        components,_=cv2.findContours(light,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        if components:
            big=max(components,key=cv2.contourArea)
            corners=cv2.approxPolyDP(big,.025*cv2.arcLength(big,True),True)
            bounds=settings['board']['area_range_px']
            if len(corners)==4 and bounds[0]<=cv2.contourArea(corners)<=bounds[1]:
                board=corners+np.array([x0,y0],np.int32);scene['board_polygon_px']=board.reshape(-1,2).tolist()
    tcp=None
    if state.get('gripping') is True and state.get('eef_m') and state.get('eef_xyzw'):
        tcp=np.array(state['eef_m'])+rotate(state['eef_xyzw'],profile['tcp_offset_tool_m'])
    for kind in ('ring','peg'):
        for spec in settings.get(kind+'s',[]):
            found=_detect(spec,kind,hsv,geometry)
            if kind=='peg' and spec.get('bracket_reference'):
                registered=_registered_peg(spec,board,bgr,geometry)
                if registered is None:
                    found=[]
                else:
                    # A round patch in the peg ROI can be a ring or a gripper.
                    # The board and tip appearance establish which physical post this is.
                    interval=np.asarray(registered['tip_position_interval_base_m'])
                    center=interval.mean(axis=0)
                    allowance=np.linalg.norm(interval[1]-interval[0])/2+.004
                    matched=[row for row in found
                             if np.linalg.norm(np.asarray(row['tip_uv_px'])-registered['tip_uv_px'])<=2.
                             and np.linalg.norm(np.asarray(row['tip_base_m'])-center)<=allowance]
                    if len(matched)==1:
                        row=matched[0]
                        registered.update(current_tip_depth_measured=True,
                                          current_tip_depth_base_m=row['tip_base_m'],
                                          current_tip_depth_inliers=row['depth_inliers'])
                    found=[registered]
            if kind=='ring' and tcp is not None and spec.get('loose_candidate'):
                projected=geometry.project(tcp)
                if projected is not None:
                    x,y=map(int,projected);extra=dict(spec,roi_px=[max(0,x-45),max(0,y-45),min(w-1,x+45),min(h-1,y+45)])
                    if extra['roi_px'][0]<extra['roi_px'][2] and extra['roi_px'][1]<extra['roi_px'][3]:
                        fresh=_detect(extra,kind,hsv,geometry)
                        near=[r for r in fresh if np.min(np.linalg.norm(np.asarray(r['rim_grasp_candidates_base_m'])-tcp,axis=1))<.022]
                        if near:found=near
            if (kind=='ring' and not found and spec.get('loose_candidate') and board is not None
                    and len(settings.get('table_polygon_px',[]))>=3):
                area=np.zeros((h,w),np.uint8)
                cv2.fillPoly(area,[np.int32(settings['table_polygon_px'])],255)
                toy=np.zeros_like(area);cv2.fillPoly(toy,[board],255)
                area[cv2.dilate(toy,np.ones((15,15),np.uint8))>0]=0
                free_spec=dict(spec,roi_px=[0,0,w-1,h-1])
                found=[r for r in _detect(free_spec,kind,hsv,geometry,allowed=area)
                       if -.005<r['top_height_above_table_m']<.04]
                for r in found:r['localization']='unique_free_ring_in_current_table_image'
            if len(found)!=1:
                scene['missing'].append({'kind':kind,'id':spec['id'],'detected_candidates':len(found)})
                continue
            row=found[0]
            if kind=='peg' and 'support_uv_px' in spec:
                u,v=map(int,spec['support_uv_px'])
                if 2<=u<w-2 and 2<=v<h-2:
                    pix=np.array([(x,y) for x in range(u-2,u+3) for y in range(v-2,v+3)])
                    cloud=geometry.cloud(pix)
                    if len(cloud)>=15 and np.ptp(cloud@geometry.n)<.012:
                        row['support_z_m']=float(np.median(cloud[:,2]));row['support_uv_px']=[u,v]
            if kind=='peg':_stack_surface(row,spec,hsv,geometry)
            scene[kind+'s'].append(row)
    pegs=scene['pegs'];all_pegs=len(pegs)==len(settings.get('pegs',[])) and len(pegs)>0
    for ring in scene['rings']:
        rc=np.array(ring['center_base_m']);near=[]
        for peg in pegs:
            pt=np.array(peg['tip_base_m']);delta=rc-pt
            lateral=float(np.linalg.norm(delta-geometry.n*(delta@geometry.n)))
            if lateral<ring['inner_radius_m'] and delta@geometry.n<.012:near.append(peg)
        ring['support_status']='unknown';ring['location']='unknown'
        on_board=None if board is None else cv2.pointPolygonTest(board,tuple(ring['center_uv_px']),True)
        if len(near)==1:
            ring['peg_id']=near[0]['id'];ring['support_status']='coaxial_with_observed_post';ring['location']='on_peg'
            near[0].setdefault('observed_ring_ids',[]).append(ring['id'])
        elif (on_board is not None and on_board < -10 or
              all_pegs and all(np.linalg.norm((rc-np.array(p['tip_base_m']))@geometry.basis[:,:2])>ring['outer_radius_m']+.012 for p in pegs)):
            ring['support_status']='clear_of_observed_posts'
            if -.005<ring['top_height_above_table_m']<.04:
                ring['location']='loose'
                _local_table(ring,bgr,geometry)
                spec=next(s for s in settings['rings'] if s['id']==ring['id'])
                if spec.get('local_support_reference'):
                    from .ring_support_reference import check_reference
                    evidence=check_reference(geometry,bgr,ring,spec['local_support_reference'])
                    ring['local_support_reference_verified']=evidence['verified']
                    ring['local_support_reference_evidence']=evidence
    if tcp is not None:
        held=[]
        for r in scene['rings']:
            if r['top_height_above_table_m']<=.05 or np.min(np.linalg.norm(np.asarray(r['rim_grasp_candidates_base_m'])-tcp,axis=1))>=.022:continue
            above_tip=any(np.linalg.norm((np.asarray(r['center_base_m'])-p['tip_base_m'])@geometry.basis[:,:2])<r['outer_radius_m']
                          and np.dot(np.asarray(r['center_base_m'])-p['tip_base_m'],geometry.n)>.002 for p in pegs)
            if r['support_status']=='clear_of_observed_posts' or above_tip:held.append(r)
        if len(held)==1:
            r=held[0];scene['held_ring_id']=r['id'];r.update(location='held',peg_id=None)
    for peg in pegs:
        mounted=[r for r in scene['rings'] if r['location']=='on_peg' and r['peg_id']==peg['id']]
        peg['observed_ring_ids']=[r['id'] for r in mounted]
        if mounted:peg['support_z_m']=max(r['center_base_m'][2] for r in mounted)
        if peg.get('protected_color_face_visible') and peg.get('measurement_source') and not mounted:
            spec=next(s for s in settings['pegs'] if s['id']==peg['id'])
            ring_id=spec.get('protected_ring_id')
            if ring_id:
                scene['rings'].append({'id':ring_id,'kind':'ring','color':peg['color'],'location':'on_peg',
                    'peg_id':peg['id'],'center_uv_px':peg['support_uv_px'],'center_base_m':peg['support_center_base_m'],
                    'inner_radius_m':None,'outer_radius_m':None,'graspable':False,'normal_base':None,
                    'geometry_status':'protected_marker_visible_hole_not_measured','visible_pixels':peg['protected_color_pixels']})
                peg['observed_ring_ids']=[ring_id]
                scene['missing']=[m for m in scene['missing'] if m['id']!=ring_id]
        peg['stack_bottom_to_top']=None  # A top view alone cannot count hidden identical layers.
    return scene
