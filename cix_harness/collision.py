"""Conservative cross-arm capsule clearance from every moving-chain link.

This checks the two arms, configured tools and a held-object envelope. It is not
a mesh collision checker for the torso, table, people or unmodelled obstacles.
Capsule radii/tool geometry require site verification before armed dual use.
"""
import math
import argparse
import json
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

from .contract import vector


class ArmCollisionError(ValueError):
    pass


class PeerStateChangedError(RuntimeError):
    pass


class ClearanceUnavailableError(RuntimeError):
    pass


def _number(value, low, high, name):
    if type(value) not in (int,float) or not math.isfinite(value) or not low<=value<=high:
        raise ValueError('Invalid '+name)
    return float(value)


def validate_clearance_config(cfg, armed=False):
    if cfg is None:
        if armed:raise ValueError('Armed dual bridge requires verified inter-arm capsule geometry')
        return None
    if not isinstance(cfg,dict):raise ValueError('inter_arm_clearance must be an object')
    if type(cfg.get('verified')) is not bool or (armed and cfg['verified'] is not True):
        raise ValueError('Inter-arm capsule geometry must be explicitly verified before arming')
    _number(cfg['minimum_distance_m'],.005,.3,'minimum inter-arm distance')
    _number(cfg['max_peer_joint_drift_rad'],.001,.05,'inactive-arm joint drift')
    _number(cfg['max_joint_sample_rad'],.001,.05,'collision sampling step')
    _number(cfg.get('max_peer_gripper_drift_percent',2.),.1,10.,'inactive-gripper drift')
    for arm in ('left','right'):
        radii=cfg['link_radii_m'][arm]
        if not isinstance(radii,dict) or not radii:
            raise ValueError('Explicit radii are required for every '+arm+' arm link')
        for link,radius in radii.items():
            if not isinstance(link,str) or not link.startswith(arm+'_'):
                raise ValueError('Capsule link must name its own arm')
            _number(radius,.005,.3,'link radius')
        tools=cfg['tool_capsules'][arm]
        if not isinstance(tools,list) or not 1<=len(tools)<=8:
            raise ValueError('Describe the actual gripper/fingers with 1..8 tool capsules')
        for tool in tools:
            for key in ('start_m','end_m'):
                values=vector(tool[key],3,'tool capsule '+key)
                if math.sqrt(sum(v*v for v in values))>1:
                    raise ValueError('Tool capsule offset exceeds one metre')
            _number(tool['radius_m'],.005,.3,'tool capsule radius')
        _number(cfg['held_object_radius_m'][arm],.005,.3,'held-object envelope radius')
    return cfg


def arm_chain_links(robot_description,arm):
    """Enumerate the exact chain without ROS/KDL imports for an offline template."""
    if arm not in ('left','right'):raise ValueError('Unknown arm')
    root=ET.fromstring(robot_description)
    parents={}
    for joint in root.findall('joint'):
        parent=joint.find('parent');child=joint.find('child')
        if parent is None or child is None:raise ValueError('URDF joint is missing parent or child')
        child_name=child.get('link')
        if child_name in parents:raise ValueError('URDF link has multiple parents')
        parents[child_name]=(parent.get('link'),joint.get('type'),joint.get('name'))
    chain=[];link=arm+'_gripper';seen=set()
    while link!='base':
        if link in seen or link not in parents:raise ValueError('No unambiguous base-to-gripper chain for '+arm)
        seen.add(link);parent,kind,name=parents[link]
        chain.append((link,kind,name));link=parent
    chain.reverse()
    moving=[index for index,(_,kind,_) in enumerate(chain) if kind!='fixed']
    if len(moving)!=7:raise ValueError('Expected seven moving arm joints in robot_description')
    links=[name for name,_,_ in chain[moving[0]:]]
    if any(not name.startswith(arm+'_') for name in links):raise ValueError('Unexpected non-arm child link')
    return links


def build_collision_candidate(robot_description,link_radius_m=.15,tool_radius_m=.08,
                              tool_length_m=.20,held_object_radius_m=.08):
    """Uniform conservative estimates for preview, never measured/verified radii.

    The URDF supplies actual link names and kinematics, not these dimensions.
    Check physical link envelopes and installed long fingers before verification.
    """
    _number(tool_length_m,.01,.8,'candidate installed tool length')
    cfg={'verified':False,'minimum_distance_m':.04,'max_peer_joint_drift_rad':.015,
         'max_joint_sample_rad':.02,'max_peer_gripper_drift_percent':2.,
         'link_radii_m':{arm:{link:link_radius_m for link in arm_chain_links(robot_description,arm)} for arm in ('left','right')},
         'tool_capsules':{arm:[{'start_m':[0.,0.,0.],'end_m':[0.,0.,tool_length_m],'radius_m':tool_radius_m}] for arm in ('left','right')},
         'held_object_radius_m':{arm:held_object_radius_m for arm in ('left','right')},
         'geometry_source':'robot_description link names with UNMEASURED uniform capsule dimensions',
         'notes':'PREVIEW CANDIDATE ONLY. Uniform radii and tool +Z envelope are estimates, not site measurements. Verify actual URDF link coverage, installed tool direction/length, open-finger envelope and held-object bound before marking verified. Does not cover torso/table/arbitrary scene obstacles.'}
    return validate_clearance_config(cfg,False)


def _dot(a,b):return sum(x*y for x,y in zip(a,b))
def _sub(a,b):return [x-y for x,y in zip(a,b)]
def _at(a,d,t):return [x+t*y for x,y in zip(a,d)]
def _clamp(v):return min(1.,max(0.,v))


def segment_distance(a,b,c,d):
    """Distance between finite 3-D segments, including points and parallel axes."""
    u,v,w=_sub(b,a),_sub(d,c),_sub(a,c)
    aa,bb,cc=_dot(u,u),_dot(u,v),_dot(v,v)
    dd,ee=_dot(u,w),_dot(v,w)
    if aa<=1e-18 and cc<=1e-18:return math.dist(a,c)
    if aa<=1e-18:s,t=0.,_clamp(ee/cc)
    elif cc<=1e-18:s,t=_clamp(-dd/aa),0.
    else:
        denominator=aa*cc-bb*bb
        s=_clamp((bb*ee-cc*dd)/denominator) if denominator>1e-18 else 0.
        t=(bb*s+ee)/cc
        if t<0.:t=0.;s=_clamp(-dd/aa)
        elif t>1.:t=1.;s=_clamp((bb-dd)/aa)
    return math.dist(_at(a,u,s),_at(c,v,t))


@dataclass(frozen=True)
class Capsule:
    name: str
    start: tuple
    end: tuple
    radius: float


def check_capsules(active,peer,minimum_distance,padding=0.,active_padding=None,peer_padding=None):
    if not active or not peer:raise ClearanceUnavailableError('Missing full-arm capsule coverage')
    for capsules,bounds in ((active,active_padding),(peer,peer_padding)):
        if bounds is not None and (set(bounds)!={c.name for c in capsules} or
                any(not math.isfinite(v) or v<0 for v in bounds.values())):
            raise ClearanceUnavailableError('Incomplete capsule displacement bounds')
    minimum=math.inf
    for a in active:
        for b in peer:
            separation=segment_distance(a.start,a.end,b.start,b.end)-a.radius-b.radius-padding
            separation-=(active_padding or {}).get(a.name,0.)+(peer_padding or {}).get(b.name,0.)
            if not math.isfinite(separation):raise ClearanceUnavailableError('Nonfinite capsule geometry')
            minimum=min(minimum,separation)
            if separation<minimum_distance:
                raise ArmCollisionError('Inter-arm clearance {:.4f} m below {:.4f} m: {} / {}'.format(
                    separation,minimum_distance,a.name,b.name))
    return minimum


class KDLArmCapsules:
    """FK at every arm-chain link, plus explicitly measured local tool capsules."""
    def __init__(self,arm,cfg):
        import sys
        from pathlib import Path
        noetic='/opt/ros/noetic/lib/python3/dist-packages'
        if Path(noetic,'urdf_parser_py').is_dir() and noetic not in sys.path[:1]:sys.path.insert(0,noetic)
        import PyKDL as kdl
        from urdf_parser_py.urdf import URDF
        from .upstream.kdl_parser import kdl_tree_from_urdf_model
        robot=URDF.from_parameter_server(key='robot_description')
        self.kdl=kdl;self.arm=arm;self.cfg=cfg
        self.chain=kdl_tree_from_urdf_model(robot).getChain('base',arm+'_gripper')
        moving=[i for i in range(self.chain.getNrOfSegments())
                if self.chain.getSegment(i).getJoint().getType()!=getattr(kdl.Joint,'None')]
        if len(moving)!=7:raise ClearanceUnavailableError('Collision model expects seven arm joints')
        self.names=[self.chain.getSegment(i).getJoint().getName() for i in moving]
        self.first=moving[0]
        self.links=[self.chain.getSegment(i).getName() for i in range(self.first,self.chain.getNrOfSegments())]
        if any(not link.startswith(arm+'_') for link in self.links):
            raise ClearanceUnavailableError('Unexpected non-arm link in collision chain')
        if set(cfg['link_radii_m'][arm])!=set(self.links):
            raise ClearanceUnavailableError('Capsule radii must cover exactly these {} links: {}'.format(arm,','.join(self.links)))
        self.fk=kdl.ChainFkSolverPos_recursive(self.chain)
        self.limits={name:(robot.joint_map[name].limit.lower,robot.joint_map[name].limit.upper) for name in self.names}
        # Sum of rigid offset lengths bounds point displacement from any joint.
        lengths=[float(self.chain.getSegment(i).getFrameToTip().p.Norm()) for i in range(self.chain.getNrOfSegments())]
        offsets=sum(lengths)
        tool_extent=max(math.sqrt(sum(x*x for x in tool[key])) for tool in cfg['tool_capsules'][arm] for key in ('start_m','end_m'))
        self.reach_bound=float(offsets+tool_extent)
        # A joint rotates only its descendants. Its own origin and preceding
        # base/arm offsets do not contribute to the lever arm. Triangle bounds
        # on all downstream offsets cover every capsule endpoint at any pose.
        self.joint_reach_bounds={name:sum(lengths[index+1:])+tool_extent
                                 for name,index in zip(self.names,moving)}
        self.capsule_reach_bounds={}
        for i in range(self.first,self.chain.getNrOfSegments()):
            self.capsule_reach_bounds[self.chain.getSegment(i).getName()]={
                name:sum(lengths[j+1:i+1]) if j<i else 0. for name,j in zip(self.names,moving)}
        for i,tool in enumerate(cfg['tool_capsules'][arm]):
            extent=max(math.sqrt(sum(x*x for x in tool[key])) for key in ('start_m','end_m'))
            self.capsule_reach_bounds[arm+'_installed_tool_'+str(i)]={
                name:sum(lengths[j+1:])+extent for name,j in zip(self.names,moving)}
        if not math.isfinite(self.reach_bound) or self.reach_bound<=0:
            raise ClearanceUnavailableError('Invalid URDF motion-radius bound')

    def displacement_bound(self,deltas):
        if set(deltas)!=set(self.names):raise ClearanceUnavailableError('Incomplete displacement joint state')
        return sum(_number(abs(deltas[n]),0,2*math.pi,'joint displacement')*self.joint_reach_bounds[n]
                   for n in self.names)

    def capsule_displacement_bounds(self,deltas):
        self.displacement_bound(deltas)  # Validate all seven joint deltas first.
        return {cap:sum(abs(deltas[n])*bounds[n] for n in self.names)
                for cap,bounds in self.capsule_reach_bounds.items()}

    def capsules(self,joints,holding=False):
        if set(joints)!=set(self.names):raise ClearanceUnavailableError('Collision joint state is incomplete')
        q=self.kdl.JntArray(len(self.names))
        for i,name in enumerate(self.names):
            low,high=self.limits[name]
            q[i]=_number(joints[name],low,high,'collision joint')
        origins=[(0.,0.,0.)];frames=[]
        for index in range(self.chain.getNrOfSegments()):
            frame=self.kdl.Frame()
            if self.fk.JntToCart(q,frame,index+1)<0:raise ClearanceUnavailableError('Link FK failed')
            origins.append(tuple(float(frame.p[i]) for i in range(3)));frames.append(frame)
        capsules=[Capsule(self.chain.getSegment(i).getName(),origins[i],origins[i+1],
                          self.cfg['link_radii_m'][self.arm][self.chain.getSegment(i).getName()])
                  for i in range(self.first,self.chain.getNrOfSegments())]
        for index,tool in enumerate(self.cfg['tool_capsules'][self.arm]):
            ends=[]
            for key in ('start_m','end_m'):
                p=frames[-1]*self.kdl.Vector(*tool[key]);ends.append(tuple(float(p[i]) for i in range(3)))
            # Inflate the entire tool envelope when holding; no guessed object centre.
            radius=tool['radius_m']+(self.cfg['held_object_radius_m'][self.arm] if holding else 0.)
            capsules.append(Capsule(self.arm+'_installed_tool_'+str(index),ends[0],ends[1],radius))
        return capsules


def state_joints(state):
    names=state.get('joint_names',[]);values=state.get('joint_rad',[])
    if not names or len(names)!=len(set(names)) or len(names)!=len(values):
        raise PeerStateChangedError('Incomplete peer joint telemetry')
    vector(values,len(names),'peer joint telemetry')
    return dict(zip(names,values))


class InterArmGuard:
    def __init__(self,cfg,models):
        self.cfg=validate_clearance_config(cfg)
        if self.cfg is None:raise ClearanceUnavailableError('No inter-arm clearance configuration')
        if set(models)!={'left','right'}:raise ClearanceUnavailableError('Both arm collision models are required')
        self.models=models

    def check_peer(self,reference,current):
        a,b=state_joints(reference),state_joints(current)
        if set(a)!=set(b) or max(abs(a[n]-b[n]) for n in a)>self.cfg['max_peer_joint_drift_rad']:
            raise PeerStateChangedError('Inactive arm moved since observation; reobserve before motion')
        if type(current.get('gripping')) is not bool or current['gripping']!=reference.get('gripping'):
            raise PeerStateChangedError('Inactive arm occupancy changed or is unknown')
        x=reference.get('gripper_percent');y=current.get('gripper_percent')
        if type(x) not in (int,float) or type(y) not in (int,float) or not math.isfinite(x+y):
            raise PeerStateChangedError('Inactive gripper position is unknown')
        if abs(x-y)>self.cfg.get('max_peer_gripper_drift_percent',2.):
            raise PeerStateChangedError('Inactive gripper moved since observation')
        return b

    def current_clearance(self,arm,active_state,peer_state):
        peer='right' if arm=='left' else 'left'
        if type(active_state.get('gripping')) is not bool or type(peer_state.get('gripping')) is not bool:
            raise PeerStateChangedError('Arm occupancy is unknown')
        return check_capsules(self.models[arm].capsules(state_joints(active_state),active_state['gripping']),
                              self.models[peer].capsules(state_joints(peer_state),peer_state['gripping']),
                              self.cfg['minimum_distance_m'])

    def path_clearance(self,arm,initial,paths,peer_reference,peer_current):
        peer='right' if arm=='left' else 'left'
        self.check_peer(peer_reference,peer_current)
        # Drift is bounded about the observation, not about a later shifted pose.
        peer_joints=state_joints(peer_reference)
        peer_model=self.models[peer];model=self.models[arm]
        peer_capsules=peer_model.capsules(peer_joints,peer_current['gripping'])
        # Protect the allowed peer drift as well as the intervals between samples.
        def displacement(model,deltas):
            method=getattr(type(model),'displacement_bound',None)
            if callable(method):return method(model,deltas)
            return sum(abs(v) for v in deltas.values())*model.reach_bound
        def capsule_bounds(model,deltas,capsules):
            method=getattr(type(model),'capsule_displacement_bounds',None)
            if callable(method):return method(model,deltas)
            return {c.name:displacement(model,deltas) for c in capsules}
        peer_bounds=capsule_bounds(peer_model,{n:self.cfg['max_peer_joint_drift_rad'] for n in peer_joints},peer_capsules)
        peer_padding=max(peer_bounds.values())
        previous=dict(initial);minimum=math.inf
        configurations=[solution for path in paths if path is not None for solution,_ in path]
        if not configurations:configurations=[previous]
        for target in configurations:
            if set(target)!=set(previous):raise ClearanceUnavailableError('Incomplete planned collision joint state')
            count=max(1,math.ceil(max(abs(target[n]-previous[n]) for n in target)/self.cfg['max_joint_sample_rad']))
            start=dict(previous)
            for index in range(1,count+1):
                q={n:start[n]+(target[n]-start[n])*index/count for n in start}
                for endpoint in (previous,q):
                    # A pick may acquire an object halfway through: use its envelope throughout.
                    active=model.capsules(endpoint,True)
                    swept=capsule_bounds(model,{n:.5*(q[n]-previous[n]) for n in q},active)
                    distance=check_capsules(active,peer_capsules,self.cfg['minimum_distance_m'],
                                            active_padding=swept,peer_padding=peer_bounds)
                    minimum=min(minimum,distance)
                previous=q
        return {'model':'urdf_chain_capsules','checked':True,'minimum_clearance_m':minimum,
                'peer_drift_padding_m':peer_padding,'swept_intervals_checked':True,
                'padding_scope':'per_capsule_descendant_joints',
                'limitations':'Cross-arm links/tools only; not torso, table or arbitrary scene obstacles'}


def main():
    parser=argparse.ArgumentParser(description='Build unverified cross-arm capsule geometry from an actual URDF; no motor operations.')
    source=parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--robot-description-xml','--urdf',dest='robot_description_xml',help='Previously saved complete URDF XML for offline generation')
    source.add_argument('--from-ros','--ros',dest='from_ros',action='store_true',help='Explicit read-only robot_description parameter lookup')
    parser.add_argument('--dual-config',help='Optional existing dual candidate to copy with the generated clearance configuration')
    parser.add_argument('--output',required=True)
    parser.add_argument('--candidate-link-radius',type=float,default=.15)
    parser.add_argument('--candidate-tool-radius',type=float,default=.08)
    parser.add_argument('--candidate-tool-length',type=float,default=.20)
    parser.add_argument('--candidate-held-radius',type=float,default=.08)
    for arm in ('left','right'):
        parser.add_argument('--'+arm+'-tool-start',type=float,nargs=3,metavar=('X','Y','Z'))
        parser.add_argument('--'+arm+'-tool-end',type=float,nargs=3,metavar=('X','Y','Z'))
        parser.add_argument('--'+arm+'-tool-radius',type=float)
    args=parser.parse_args()
    if args.from_ros:
        import rospy
        rospy.init_node('cix_capsule_candidate_readonly',anonymous=True,disable_signals=True)
        try:description=rospy.get_param('robot_description')
        finally:rospy.signal_shutdown('read-only URDF lookup complete')
    else:description=Path(args.robot_description_xml).read_text(encoding='utf-8')
    result=build_collision_candidate(description,args.candidate_link_radius,args.candidate_tool_radius,
                                     args.candidate_tool_length,args.candidate_held_radius)
    for arm in ('left','right'):
        tool=result['tool_capsules'][arm][0]
        for key in ('start','end'):
            value=getattr(args,arm+'_tool_'+key)
            if value is not None:tool[key+'_m']=value
        radius=getattr(args,arm+'_tool_radius')
        if radius is not None:tool['radius_m']=radius
    validate_clearance_config(result,False)
    if args.dual_config:
        from .dual_bridge import build_dual_candidate
        original=json.loads(Path(args.dual_config).read_text(encoding='utf-8'))
        result=build_dual_candidate(original['arms']['left'],original['arms']['right'],
                                    original['motion_budget'],original['lease_seconds'],result)
    output=Path(args.output)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as target:json.dump(result,target,ensure_ascii=False,indent=2)
    print('Wrote unverified read-only capsule candidate:',output)


if __name__=='__main__':main()
