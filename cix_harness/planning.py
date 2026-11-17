"""Bounded grasp-orientation candidates; every candidate needs full path checks."""
import copy
import math
from .pi05_contract import multiply,quaternion


def candidate_profiles(profile,state,skill,options):
    if skill not in ('pick','move_to') or state.get('gripping'):
        yield 0.,profile
        return
    values=options.get('grasp_yaw_offsets_rad',[0.])
    if (not isinstance(values,list) or not 1<=len(values)<=8 or
            any(type(v) not in (float,int) or not math.isfinite(v) or abs(v)>math.pi for v in values)):
        raise ValueError('Grasp yaw candidates require 1..8 finite angles within pi')
    current=quaternion(state['eef_xyzw']);base=quaternion(profile['tool_orientation_xyzw'])
    candidates=[]
    for yaw in dict.fromkeys(values):
        p=copy.deepcopy(profile)
        p['tool_orientation_xyzw']=multiply([0,0,math.sin(yaw/2),math.cos(yaw/2)],base)
        angle=2*math.acos(min(1.,abs(sum(x*y for x,y in zip(current,p['tool_orientation_xyzw'])))))
        candidates.append((angle,yaw,p))
    # A symmetric cube allows several yaw choices; try the least wrist rotation first.
    for _,yaw,p in sorted(candidates,key=lambda x:x[0]):yield yaw,p


def stage_phase(stages,index,skill):
    stage=stages[index]
    if stage['kind']!='move':return stage['kind']
    earlier=stages[:index]
    closed=any(s.get('kind')=='gripper' and s.get('open') is False for s in earlier)
    opened=any(s.get('kind')=='gripper' and s.get('open') is True for s in earlier)
    if skill=='pick' and closed:return 'lift'
    if skill in ('place','stack') and opened:return 'retreat'
    if index and stages[index-1].get('kind')=='move':
        previous=stages[index-1]
        if 'tcp_m' in stage and 'tcp_m' in previous:
            if math.dist(stage['tcp_m'],previous['tcp_m'])<1e-6:return 'orientation'
            if math.dist(stage['tcp_m'][:2],previous['tcp_m'][:2])>1e-6:return 'transit'
    if skill=='pick' and opened:return 'approach'
    if skill in ('place','stack') and index+1<len(stages) and stages[index+1].get('kind')=='gripper':return 'placement'
    return 'clearance'
