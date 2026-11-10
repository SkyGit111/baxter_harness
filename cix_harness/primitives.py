"""Deterministic tabletop skill plans, in calibrated Baxter base / TCP coordinates.

Sequencing references Baxter-claw bridge/primitives.py (MIT); no hardcoded table Z.
"""
import math
from .contract import vector
from .pi05_contract import quaternion, rotate

SKILLS = ('pick', 'place', 'stack', 'move_to', 'lift', 'retreat', 'open', 'close', 'home', 'clear_view')


class TargetDepthError(ValueError):
    """No motion occurred; a fresh image and a different interior pixel can help."""

def number(value, low, high, name):
    if type(value) not in (float, int) or not math.isfinite(value) or not low <= value <= high:
        raise ValueError('Invalid '+name)
    return value

def validate_profile(p):
    if p.get('frame') != 'base': raise ValueError('Primitive frame must be base')
    quaternion(p['tool_orientation_xyzw'])
    quaternion(p['camera_to_base_xyzw'])
    vector(p['camera_to_base_translation_m'], 3, 'camera translation')
    vector(p['tcp_offset_tool_m'], 3, 'TCP offset')
    low=vector(p['tcp_workspace_min_m'],3,'workspace minimum')
    high=vector(p['tcp_workspace_max_m'],3,'workspace maximum')
    if any(a>=b for a,b in zip(low,high)): raise ValueError('Unordered workspace')
    for key,lo,hi in [('table_z_m',-2,2),('table_clearance_m',0,.1),('transit_z_m',-1,2),
                      ('approach_m',.02,.3),('lift_m',.025,.3),('grasp_inset_m',0,.1),
                      ('release_clearance_m',0,.05),('held_bottom_below_tcp_m',0,.3),
                      ('max_leg_m',.01,.5),('perception_max_age_s',1,60)]:
        number(p[key],lo,hi,key)
    if not p['table_z_m']+p['table_clearance_m'] <= p['transit_z_m'] <= high[2]:
        raise ValueError('Transit height outside workspace')
    views=p.get('observation_tcp_m',[])
    if not isinstance(views,list) or len(views)>4:raise ValueError('Expected up to four observation positions')
    for view in views:
        xyz=vector(view,3,'observation TCP')
        if any(not lo<=v<=hi for v,lo,hi in zip(xyz,low,high)) or xyz[2]<p['transit_z_m']:
            raise ValueError('Observation position must be inside workspace and at/above transit height')
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

def plan(skill,state,p,target=None,view_index=0):
    validate_profile(p)
    if skill not in SKILLS: raise ValueError('Unknown primitive')
    current=tcp_from_eef(state['eef_m'],state['eef_xyzw'],p)
    orientation=p['tool_orientation_xyzw']
    holding=state.get('gripping')
    if type(holding) is not bool: raise ValueError('Grasp telemetry unknown')
    if skill=='pick' and holding: raise ValueError('Already holding an object')
    if skill=='open' and holding: raise ValueError('Use place/stack to release a held object onto a support')
    if skill in ('place','stack') and not holding: raise ValueError('Placement requires a held object')
    stages=[]
    def move(xyz,q=None):
        xyz=vector(xyz,3,'TCP target')
        if any(not lo<=v<=hi for v,lo,hi in zip(xyz,p['tcp_workspace_min_m'],p['tcp_workspace_max_m'])):
            raise ValueError('TCP target outside calibrated workspace')
        floor=p['table_z_m']+p['table_clearance_m']
        if xyz[2]<floor: raise ValueError('TCP target below table clearance')
        stages.append({'kind':'move','tcp_m':xyz,'xyzw':q or orientation})
    def grip(opened): stages.append({'kind':'gripper','open':opened})
    if skill in ('pick','place','stack','move_to'):
        point=vector(target,3,'base target')
        if skill=='pick':point[2]-=p['grasp_inset_m']
        elif skill in ('place','stack'):
            point[2]+=p['held_bottom_below_tcp_m']+p['release_clearance_m']
        z=max(current[2],point[2]+p['approach_m'],p['transit_z_m'])
        move([current[0],current[1],z],state['eef_xyzw'])
        move([current[0],current[1],z])
        move([point[0],point[1],z])
        if skill=='pick':grip(True)
        move(point)
        if skill=='pick':
            grip(False);stages.append({'kind':'verify_grasp','held':True})
            move([point[0],point[1],max(point[2]+p['lift_m'],z)])
            stages.append({'kind':'verify_grasp','held':True})
        elif skill in ('place','stack'):
            grip(True);stages.append({'kind':'verify_grasp','held':False})
            move([point[0],point[1],z])
    elif skill=='clear_view':
        views=p.get('observation_tcp_m',[])
        if type(view_index) is not int or not 0<=view_index<len(views):
            raise ValueError('No configured observation position at this index')
        view=vector(views[view_index],3,'observation TCP')
        if math.dist(current,view)<.02:raise ValueError('Already at observation position')
        floor=p['table_z_m']+p['table_clearance_m']
        if holding and view[2]<floor+p['held_bottom_below_tcp_m']:
            raise ValueError('Held object would cross table at observation position')
        z=max(current[2],view[2],p['transit_z_m'])
        # Lift before translating. Preserve tool orientation and never operate the gripper.
        move([current[0],current[1],z],state['eef_xyzw'])
        move([view[0],view[1],z],state['eef_xyzw'])
        move(view,state['eef_xyzw'])
        if holding:stages.append({'kind':'verify_grasp','held':True})
    elif skill in ('lift','retreat'):
        move([current[0],current[1],current[2]+p['lift_m']],state['eef_xyzw'])
        if holding:stages.append({'kind':'verify_grasp','held':True})
    elif skill in ('open','close'):grip(skill=='open')
    elif skill=='home':
        if holding:raise ValueError('Home requires an empty gripper')
        home=vector(p['home_tcp_m'],3,'configured home')
        z=max(current[2],home[2],p['transit_z_m'])
        move([current[0],current[1],z],state['eef_xyzw']);move([home[0],home[1],z]);move(home)
    return stages
