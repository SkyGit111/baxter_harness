"""Compose policy deltas without multiplying their amplitude or skipping reversals."""
import math
from .contract import action7
from .pi05_contract import multiply


def euler_quaternion(rpy):
    r,p,y=[v/2 for v in rpy]
    cr,sr,cp,sp,cy,sy=math.cos(r),math.sin(r),math.cos(p),math.sin(p),math.cos(y),math.sin(y)
    return [sr*cp*cy-cr*sp*sy,cr*sp*cy+sr*cp*sy,cr*cp*sy-sr*sp*cy,cr*cp*cy+sr*sp*sy]


def quaternion_euler(q):
    x,y,z,w=q
    return [math.atan2(2*(w*x+y*z),1-2*(x*x+y*y)),
            math.asin(max(-1.,min(1.,2*(w*y-z*x)))),
            math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))]


def compose_prefix(actions, currently_open, path_limit=.12):
    if not isinstance(actions,list) or not 1 <= len(actions) <= 16:
        raise ValueError('A policy segment requires 1..16 actions')
    actions=[action7(a) for a in actions]  # Validate even the unexecuted suffix.
    xyz=[0.,0.,0.];q=[0.,0.,0.,1.];length=0.;used=0;reason='window_complete'
    previous=None
    for action in actions:
        distance=math.sqrt(sum(v*v for v in action[:3]))
        if length+distance > path_limit:
            reason='path_budget'
            break
        # A straight consolidated segment must not shortcut a reversal/turn.
        if previous is not None and distance>1e-6 and sum(a*b for a,b in zip(previous,action[:3])) < 0:
            reason='direction_change'
            break
        xyz=[a+b for a,b in zip(xyz,action[:3])]
        q=multiply(euler_quaternion(action[3:6]),q)
        length+=distance;used+=1;previous=action[:3]
        if (action[6]>=.5) != currently_open:
            reason='gripper_transition'
            break
    if not used:
        raise ValueError('First policy action exceeds the segment path budget')
    return xyz+quaternion_euler(q)+[actions[used-1][6]],used,reason,length
