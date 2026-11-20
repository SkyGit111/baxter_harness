"""Ground a held ring's clearance route in the receiving peg observation."""
import math

from .contract import vector
from .pi05_contract import quaternion,rotate
from .primitives import eef_from_tcp,number,plan,tcp_from_eef
from .ring_peg import RingGeometryError,measured


def validate(route):
    vector(route['via_tcp_base_m'],3,'ring transport waypoint in base')
    quaternion(route['via_orientation_xyzw'])
    quaternion(route['peg_orientation_xyzw'])
    number(route['above_peg_m'],.025,.20,'held ring hover height above peg')
    number(route['approach_m'],.02,.15,'held ring clearance approach')
    number(route['payload_radius_m'],.005,.15,'transported ring bounding radius around TCP')


def check_payload(point,boxes,radius):
    for box in boxes:
        if box['frame']!='base':continue
        x,y,z,w=box['xyzw']
        local=rotate([-x,-y,-z,w],[a-b for a,b in zip(point,box['xyz'])])
        distance=math.sqrt(sum(max(abs(a)-b/2,0.)**2 for a,b in zip(local,box['size_m'])))
        if distance<=radius:
            raise RingGeometryError('Held ring envelope meets scene object '+box['id'])


def hover(profile,peg):
    route=profile['ring_peg']['transport']
    tip=measured(peg,'tip_base_m',3)
    return [tip[0],tip[1],tip[2]+route['above_peg_m']]


def at_hover(profile,state,peg):
    point=tcp_from_eef(state['eef_m'],state['eef_xyzw'],profile)
    orientation=profile['ring_peg']['transport']['peg_orientation_xyzw']
    dot=abs(sum(a*b for a,b in zip(orientation,state['eef_xyzw'])))
    return math.dist(point,hover(profile,peg))<=.015 and 2*math.acos(min(1.,dot))<=.06


def stages(profile,state,peg):
    route=profile['ring_peg']['transport']
    current=dict(state)
    result=[]
    for target,orientation in [(route['via_tcp_base_m'],route['via_orientation_xyzw']),
                               (hover(profile,peg),route['peg_orientation_xyzw'])]:
        leg=dict(profile,approach_m=route['approach_m'],tool_orientation_xyzw=orientation)
        result.extend(plan('move_to',current,leg,target))
        result.append({'kind':'verify_grasp','held':True})
        current.update(eef_m=eef_from_tcp(target,orientation,leg),eef_xyzw=orientation)
    return result


def release_profile(profile,state,peg):
    if not at_hover(profile,state,peg):
        raise RingGeometryError('Transport held ring to the matching peg with move_to and reobserve before release')
    route=profile['ring_peg']['transport']
    return dict(profile,tool_orientation_xyzw=route['peg_orientation_xyzw'],approach_m=route['approach_m'])
