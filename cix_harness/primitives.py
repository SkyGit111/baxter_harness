"""Deterministic tabletop skill plans, in calibrated Baxter base / TCP coordinates.

Sequencing references Baxter-claw bridge/primitives.py (MIT); no hardcoded table Z.
"""
import math
from .contract import vector
from .pi05_contract import quaternion, rotate

SKILLS = ('pick', 'place', 'stack', 'move_to', 'lift', 'retreat', 'open', 'close', 'home', 'clear_view')


class TargetDepthError(ValueError):
    """No motion occurred; a fresh image and a different interior pixel can help."""


class TargetWorkspaceError(ValueError):
    """An explicitly selected contact target is outside the calibrated TCP region."""


def table_height(x,y,p):
    """Base Z of the support surface; a plane replaces the legacy scalar height."""
    x,y=vector([x,y],2,'table XY')
    plane=p.get('table_plane')
    if plane is None:return number(p['table_z_m'],-2,2,'table_z_m')
    if not isinstance(plane,dict):raise ValueError('Table plane must be an object')
    n=vector(plane.get('normal'),3,'table plane normal')
    d=plane.get('offset')
    if type(d) not in (float,int) or not math.isfinite(d):raise ValueError('Invalid table plane offset')
    length=math.hypot(*n)
    if length<1e-9 or n[2]/length<.5:
        raise ValueError('Table plane must face upward and be less than 60 degrees from horizontal')
    z=-(n[0]*x+n[1]*y+d)/n[2]
    if not math.isfinite(z):raise ValueError('Invalid table plane height')
    return z


def _table_polygon(p):
    polygon=p.get('table_polygon_base_xy')
    if polygon is None:return None
    if not isinstance(polygon,list) or not 3<=len(polygon)<=32:
        raise ValueError('Table polygon requires 3 to 32 convex perimeter vertices')
    polygon=[vector(v,2,'table polygon vertex') for v in polygon]
    if len(set(map(tuple,polygon)))!=len(polygon):raise ValueError('Repeated table polygon vertex')
    area=sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(polygon,polygon[1:]+polygon[:1]))
    if not math.isfinite(area) or abs(area)<1e-8:raise ValueError('Degenerate table polygon')
    sign=1 if area>0 else -1
    for a,b in zip(polygon,polygon[1:]+polygon[:1]):
        if math.dist(a,b)<1e-6:raise ValueError('Degenerate table polygon edge')
        if any(sign*((b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])) < -1e-10 for c in polygon):
            raise ValueError('Table polygon must be convex and ordered around its perimeter')
    return polygon


def workspace_polygon(p):
    """Convex intersection of the table footprint and the TCP rectangular bounds."""
    low=vector(p['tcp_workspace_min_m'],3,'workspace minimum')
    high=vector(p['tcp_workspace_max_m'],3,'workspace maximum')
    if any(a>=b for a,b in zip(low,high)):raise ValueError('Unordered workspace')
    polygon=_table_polygon(p)
    if polygon is None:return [[low[0],low[1]],[high[0],low[1]],[high[0],high[1]],[low[0],high[1]]]
    for axis,bound,sign in ((0,low[0],1),(0,high[0],-1),(1,low[1],1),(1,high[1],-1)):
        clipped=[]
        for a,b in zip(polygon,polygon[1:]+polygon[:1]):
            ai=sign*(a[axis]-bound)>=0;bi=sign*(b[axis]-bound)>=0
            if ai:clipped.append(a)
            if ai!=bi:
                f=(bound-a[axis])/(b[axis]-a[axis])
                clipped.append([a[j]+f*(b[j]-a[j]) for j in (0,1)])
        polygon=clipped
        if not polygon:break
    # Clipping through a vertex may produce adjacent duplicates.
    result=[]
    for v in polygon:
        if not result or math.dist(result[-1],v)>1e-9:result.append(v)
    if len(result)>1 and math.dist(result[0],result[-1])<1e-9:result.pop()
    if len(result)<3 or abs(sum(a[0]*b[1]-b[0]*a[1] for a,b in zip(result,result[1:]+result[:1])))<1e-8:
        raise ValueError('Table polygon and TCP workspace have no usable intersection')
    return result


def validate_table_geometry(p):
    table_height(0,0,p)
    polygon=workspace_polygon(p)
    for x,y in polygon:
        if not -2<=table_height(x,y,p)<=2:raise ValueError('Table plane height outside supported base range')
    return p


def check_tcp(xyz,p,selected_target=False,held_bottom_m=0.):
    xyz=vector(xyz,3,'TCP target')
    floor=table_height(xyz[0],xyz[1],p)+p['table_clearance_m']+number(held_bottom_m,0,.3,'held bottom offset')
    # Table violations remain hard failures even if the same point is also
    # outside the workspace. Retargeting must not disguise a contact-boundary error.
    if xyz[2]<floor-1e-9:raise ValueError('TCP or held object below table clearance')
    polygon=_table_polygon(p)
    outside=False
    if polygon is not None:
        crosses=[(b[0]-a[0])*(xyz[1]-a[1])-(b[1]-a[1])*(xyz[0]-a[0]) for a,b in zip(polygon,polygon[1:]+polygon[:1])]
        outside=not(all(c>=-1e-10 for c in crosses) or all(c<=1e-10 for c in crosses))
        corridor=p.get('empty_entry_corridor')
        if outside and corridor is not None and not selected_target and not held_bottom_m:
            start,end=corridor['start_xy_m'],corridor['end_xy_m']
            dx,dy=end[0]-start[0],end[1]-start[1]
            f=max(0.,min(1.,((xyz[0]-start[0])*dx+(xyz[1]-start[1])*dy)/(dx*dx+dy*dy)))
            outside=(math.dist(xyz[:2],[start[0]+f*dx,start[1]+f*dy])>corridor['radius_m'] or
                     not corridor['z_range_m'][0]<=xyz[2]<=corridor['z_range_m'][1])
    if outside or any(not lo<=v<=hi for v,lo,hi in zip(xyz,p['tcp_workspace_min_m'],p['tcp_workspace_max_m'])):
        error=TargetWorkspaceError if selected_target else ValueError
        raise error('TCP target outside calibrated workspace; select a visible in-workspace target or replan/stop')
    return xyz


def transit_height(points,p,holding=False):
    """A plane is affine, so segment endpoint maxima bound every connecting leg."""
    margin=p['table_clearance_m']+p['approach_m']+(p['held_bottom_below_tcp_m'] if holding else 0.)
    return max(p['transit_z_m'],*(table_height(v[0],v[1],p)+margin for v in points))


def check_stack_support(point,p,*,at_release=False):
    support=p.get('stack_support')
    if support is None:return
    centre=vector(support['top_center_m'],3,'stack support top centre')
    radius=number(support['center_radius_m'],.003,.1,'stack support centre radius')
    ztol=number(support['top_tolerance_m'],.001,.03,'stack support top tolerance')
    if math.dist(point[:2],centre[:2])>radius:
        raise TargetWorkspaceError('Stack target outside the measured support centre')
    if at_release:
        gap=point[2]-p['held_bottom_below_tcp_m']-centre[2]
        if gap < 0 or gap>p['release_clearance_m']+ztol:
            raise ValueError('Held object outside the measured stacking release height')
    elif abs(point[2]-centre[2])>ztol:
        raise TargetWorkspaceError('Stack target depth disagrees with the measured support top')


def _slerp(a,b,f):
    dot=sum(x*y for x,y in zip(a,b))
    if dot<0:b=[-v for v in b];dot=-dot
    dot=min(1.,dot)
    if dot>.9995:q=[x+f*(y-x) for x,y in zip(a,b)]
    else:
        angle=math.acos(dot);s=math.sin(angle)
        q=[(math.sin((1-f)*angle)*x+math.sin(f*angle)*y)/s for x,y in zip(a,b)]
    norm=math.sqrt(sum(v*v for v in q))
    return [v/norm for v in q]


def number(value, low, high, name):
    if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('Invalid '+name)
    return value

def validate_profile(p):
    if 'ring_peg' in p:
        from .ring_peg import validate_settings
        validate_settings(p['ring_peg'])
    if p.get('frame') != 'base': raise ValueError('Primitive frame must be base')
    quaternion(p['tool_orientation_xyzw'])
    quaternion(p['camera_to_base_xyzw'])
    vector(p['camera_to_base_translation_m'], 3, 'camera translation')
    vector(p['tcp_offset_tool_m'], 3, 'TCP offset')
    low=vector(p['tcp_workspace_min_m'],3,'workspace minimum')
    high=vector(p['tcp_workspace_max_m'],3,'workspace maximum')
    if any(a>=b for a,b in zip(low,high)): raise ValueError('Unordered workspace')
    validate_table_geometry(p)
    if p.get('approach_path','staged') not in ('staged','direct_hover'):
        raise ValueError('Unknown primitive approach path')
    for key,lo,hi in [('table_z_m',-2,2),('table_clearance_m',0,.1),('transit_z_m',-1,2),
                      ('approach_m',.02,.3),('lift_m',.025,.3),('grasp_inset_m',0,.1),
                      ('release_clearance_m',0,.05),('held_bottom_below_tcp_m',0,.3),
                      ('max_leg_m',.01,.5),('perception_max_age_s',1,60)]:
        if key=='table_z_m' and key not in p and p.get('table_plane') is not None:continue
        number(p[key],lo,hi,key)
    if 'contact_band_m' in p:
        number(p['contact_band_m'],.01,p['approach_m'],'contact band')
    if 'clear_view_initial_lift_m' in p:
        number(p['clear_view_initial_lift_m'],.03,.30,'clear-view initial lift')
    if 'empty_entry_corridor' in p:
        corridor=p['empty_entry_corridor']
        if not isinstance(corridor,dict) or not isinstance(corridor.get('source'),str) or not corridor['source'].strip():
            raise ValueError('Empty entry corridor needs current scene evidence')
        start=vector(corridor['start_xy_m'],2,'entry start');end=vector(corridor['end_xy_m'],2,'entry end')
        zrange=vector(corridor['z_range_m'],2,'entry height range')
        if math.dist(start,end)<.005 or zrange[0]>=zrange[1]:
            raise ValueError('Invalid empty entry corridor dimensions')
        number(corridor['radius_m'],.005,.10,'entry corridor radius')
        if 'table_entry_m' in corridor:
            check_tcp(corridor['table_entry_m'],p,selected_target=True)
    if 'stack_support' in p:
        support=p['stack_support']
        if not isinstance(support,dict) or not isinstance(support.get('source'),str) or not support['source'].strip():
            raise ValueError('Stack support requires measured top geometry and source')
        check_stack_support(check_tcp(support['top_center_m'],p),p)
    minimum_floor=min(table_height(x,y,p) for x,y in workspace_polygon(p))+p['table_clearance_m']
    if not minimum_floor <= p['transit_z_m'] <= high[2]:
        raise ValueError('Transit height outside workspace')
    views=p.get('observation_tcp_m',[])
    if not isinstance(views,list) or len(views)>4:raise ValueError('Expected up to four observation positions')
    for view in views:
        xyz=vector(view,3,'observation TCP')
        check_tcp(xyz,p)
        if any(not lo<=v<=hi for v,lo,hi in zip(xyz,low,high)) or xyz[2]<p['transit_z_m']:
            raise ValueError('Observation position must be inside workspace and at/above transit height')
    if p.get('home_tcp_m') is not None:check_tcp(p['home_tcp_m'],p)
    return p

def tcp_from_eef(xyz,q,p):
    return [a+b for a,b in zip(xyz,rotate(q,p['tcp_offset_tool_m']))]

def eef_from_tcp(xyz,q,p):
    return [a-b for a,b in zip(xyz,rotate(q,p['tcp_offset_tool_m']))]

def ground_pixel(uv,depth,intrinsics,p):
    """Median aligned depth in a small patch; reject holes and depth discontinuities."""
    import numpy as np
    u,v=vector(uv,2,'normalized image point')
    number(u,0,1,'u');number(v,0,1,'v')
    h,w=depth.shape
    x=int(round(u*(w-1)));y=int(round(v*(h-1)))
    patch=depth[max(0,y-2):min(h,y+3),max(0,x-2):min(w,x+3)]
    good=patch[np.isfinite(patch)&(patch>.1)&(patch<3)]
    if good.size<9: raise TargetDepthError('Insufficient valid aligned depth at target; select another interior point')
    z=float(np.median(good))
    if float(np.percentile(good,90)-np.percentile(good,10))>.025:
        raise TargetDepthError('Target point crosses a depth discontinuity; choose object interior')
    # Use RealSense deprojection, including the actual distortion model.
    import pyrealsense2 as rs
    camera=rs.rs2_deproject_pixel_to_point(intrinsics,[x,y],z)
    return [float(a+b) for a,b in zip(rotate(p['camera_to_base_xyzw'],camera),p['camera_to_base_translation_m'])]

def plan(skill,state,p,target=None,view_index=0,*,ring_contact_tcp=None):
    validate_profile(p)
    if skill not in SKILLS: raise ValueError('Unknown primitive')
    current=tcp_from_eef(state['eef_m'],state['eef_xyzw'],p)
    check_tcp(current,p)
    # A pick may select a different yaw. Carry its measured holding orientation
    # through placement instead of rotating back to the default grasp profile.
    orientation=(quaternion(state['eef_xyzw']) if skill in ('place','stack')
                 else p['tool_orientation_xyzw'])
    holding=state.get('gripping')
    if type(holding) is not bool: raise ValueError('Grasp telemetry unknown')
    if skill=='pick' and holding: raise ValueError('Already holding an object')
    if skill=='open' and holding: raise ValueError('Use place/stack to release a held object onto a support')
    if skill in ('place','stack') and not holding: raise ValueError('Placement requires a held object')
    stages=[];last=current;last_q=quaternion(state['eef_xyzw']);planned_holding=holding
    direct=p.get('approach_path')=='direct_hover'
    def move(xyz,q=None,phase=None):
        nonlocal last,last_q
        q=quaternion(q or orientation)
        xyz=check_tcp(xyz,p,held_bottom_m=p['held_bottom_below_tcp_m'] if planned_holding else 0.)
        angle=2*math.acos(min(1.,abs(sum(a*b for a,b in zip(last_q,q)))))
        # This bound includes TCP rotation about a nonzero tool offset. Each
        # generated segment remains within the EEF travel budget, not just XY.
        bound=math.dist(last,xyz)+math.sqrt(sum(v*v for v in p['tcp_offset_tool_m']))*angle
        if direct and math.dist(last,xyz)<1e-9 and angle<1e-7:
            return
        count=max(1,math.ceil(bound/p['max_leg_m']))
        start=list(last);start_q=list(last_q)
        for i in range(1,count+1):
            f=i/count;point=list(xyz) if i==count else [a+f*(b-a) for a,b in zip(start,xyz)]
            check_tcp(point,p,held_bottom_m=p['held_bottom_below_tcp_m'] if planned_holding else 0.)
            stage={'kind':'move','tcp_m':point,'xyzw':q if i==count else _slerp(start_q,q,f)}
            if phase is not None:
                stage['motion_phase']=phase
            stages.append(stage)
        last=list(xyz);last_q=list(q)
    def grip(opened):
        nonlocal planned_holding
        stages.append({'kind':'gripper','open':opened});planned_holding=not opened
    if skill in ('pick','place','stack','move_to'):
        point=vector(target,3,'base target')
        if skill=='stack':check_stack_support(point,p)
        if ring_contact_tcp is not None:
            if 'ring_peg' not in p or skill not in ('pick','place'):raise ValueError('Unexpected ring contact target')
            point=vector(ring_contact_tcp,3,'measured ring contact TCP')
        elif skill=='pick':point[2]-=p['grasp_inset_m']
        elif skill in ('place','stack'):
            point[2]+=p['held_bottom_below_tcp_m']+p['release_clearance_m']
        # Validate only the selected contact target as a recoverable perception
        # rejection. Generated hover/retreat waypoints retain hard validation.
        point=check_tcp(point,p,selected_target=True)
        z=max(point[2]+p['approach_m'],transit_height([current,point],p,holding))
        if direct:
            # Start all lateral motion above the transit plane. When already
            # higher, approach the hover point and turn in the same checked leg.
            entry=None
            if skill=='pick' and 'table_entry_m' in p.get('empty_entry_corridor',{}):
                try:check_tcp(current,p,selected_target=True)
                except TargetWorkspaceError:entry=p['empty_entry_corridor']['table_entry_m']
            if entry is not None:
                move([entry[0],entry[1],max(z,entry[2])],state['eef_xyzw'],'clearance')
            elif current[2]<z:
                move([current[0],current[1],z],state['eef_xyzw'],'clearance')
            move([point[0],point[1],z],phase='transit')
        else:
            z=max(current[2],z)
            move([current[0],current[1],z],state['eef_xyzw'])
            move([current[0],current[1],z])
            move([point[0],point[1],z])
        if skill=='pick':grip(True)
        band=p.get('contact_band_m')
        # The first lift leg must fully clear the held object's bottom.
        band_z=(max(point[2]+band,table_height(*point[:2],p)+p['table_clearance_m']+
                    p['held_bottom_below_tcp_m']) if band is not None else z)
        split=direct and skill in ('pick','place','stack') and z-band_z>=.02
        if split:
            move([*point[:2],band_z],phase='transit')
        contact_phase='placement' if skill in ('place','stack') else 'approach'
        move(point,phase=contact_phase if direct else None)
        if skill=='pick':
            grip(False);stages.append({'kind':'verify_grasp','held':True})
            if split:
                move([*point[:2],band_z],phase='contact')
            lift_phase='lift_clear' if split else 'lift'
            move([point[0],point[1],max(point[2]+p['lift_m'],z)],phase=lift_phase if direct else None)
            stages.append({'kind':'verify_grasp','held':True})
        elif skill in ('place','stack'):
            grip(True);stages.append({'kind':'verify_grasp','held':False})
            if split:
                move([*point[:2],band_z],phase='contact')
            move([point[0],point[1],z],phase='retreat' if direct else None)
    elif skill=='clear_view':
        views=p.get('observation_tcp_m',[])
        if type(view_index) is not int or not 0<=view_index<len(views):
            raise ValueError('No configured observation position at this index')
        view=vector(views[view_index],3,'observation TCP')
        if math.dist(current,view)<.02:raise ValueError('Already at observation position')
        check_tcp(view,p,held_bottom_m=p['held_bottom_below_tcp_m'] if holding else 0.)
        transit=transit_height([current,view],p,holding)
        z=max(current[2],view[2],transit)
        rise=z
        if 'clear_view_initial_lift_m' in p:
            rise=max(current[2],transit,min(z,current[2]+p['clear_view_initial_lift_m']))
        # Lift before translating. Preserve tool orientation and never operate the gripper.
        move([current[0],current[1],rise],state['eef_xyzw'])
        move([view[0],view[1],z],state['eef_xyzw'])
        move(view,state['eef_xyzw'])
        if holding:stages.append({'kind':'verify_grasp','held':True})
    elif skill in ('lift','retreat'):
        move([current[0],current[1],current[2]+p['lift_m']],state['eef_xyzw'])
        if holding:stages.append({'kind':'verify_grasp','held':True})
    elif skill in ('open','close'):grip(skill=='open')
    elif skill=='home':
        if holding:raise ValueError('Home requires an empty gripper')
        if p.get('home_tcp_m') is None:raise ValueError('No configured home position; record and validate a parking pose first')
        home=vector(p['home_tcp_m'],3,'configured home')
        z=max(current[2],home[2],transit_height([current,home],p))
        move([current[0],current[1],z],state['eef_xyzw']);move([home[0],home[1],z]);move(home)
    return stages
