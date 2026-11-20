"""Locally check right-wrist calibration poses, then optionally drive and capture each stationary pose."""
import argparse
from copy import deepcopy
import itertools
import json
import math
from pathlib import Path
import time
import uuid
import xml.etree.ElementTree as ET

import numpy as np

from .handeye import angle, packed, transform
from .kinematics import ContinuousIK, IKUnreachableError
from .moveit_probe import joint_names
from .pi05_contract import multiply
from .trajectory import check_controller_feedback


def pose_msg(m):
    from geometry_msgs.msg import PoseStamped
    point=PoseStamped(); point.header.frame_id='base'
    m=packed(m)
    point.pose.position.x,point.pose.position.y,point.pose.position.z=m['xyz']
    point.pose.orientation.x,point.pose.orientation.y,point.pose.orientation.z,point.pose.orientation.w=m['xyzw']
    return point


def fk_matrix(pose):
    p,q=pose['position'],pose['orientation']
    return transform({'xyz':[p.x,p.y,p.z],'xyzw':[q.x,q.y,q.z,q.w]})


def targets(anchor,count,degrees=21,full=False):
    dirs=[d for d in itertools.product((-1,0,1), repeat=3) if 1<=sum(bool(v) for v in d)<=2]
    if not 1<=count<=len(dirs)+1:raise ValueError('Expected 1..19 target poses')
    result=[anchor.copy()]
    p=packed(anchor)
    # Start with three independent positive axes before collecting diagonals.
    primary=[(1,0,0),(0,1,0),(0,0,1)]
    order=primary+[d for d in dirs if d not in primary]
    if full:order.extend(itertools.product((-1,1),repeat=3))
    for i,d in enumerate(order if full else order[:count-1]):
        axis=np.array(d)/np.linalg.norm(d)
        h=math.radians(degrees)/2
        q=multiply(p['xyzw'],(axis*math.sin(h)).tolist()+[math.cos(h)])
        xyz=anchor[:3,3]+[.003*axis[1],.003*axis[0],.004+.001*(i%3)]
        result.append(transform({'xyz':xyz.tolist(),'xyzw':q}))
    return result


def candidate_poses(anchor):
    """Try short independent rotations first, then more directions around the local envelope."""
    yield anchor.copy()
    for degrees in (15,21,24):
        yield from targets(anchor,19,degrees,full=True)[1:]


def distinct_pose(target,accepted):
    # The RGB-D recorder's actual observed-pose novelty threshold is 12 deg.
    # Reserve a further degree for finite controller positioning repeatability.
    return all(angle(t[:3,:3].T@target[:3,:3])>=math.radians(13) for t in accepted)


def wrist_lattice(anchor):
    """Axial roll with a smaller secondary tilt and a bounded compensating TCP shift."""
    for roll in (-22,-15,-7,0,7,15,22):
        for tilt in (0,8,11,14):
            for azimuth in (range(0,360,45) if tilt else (0,)):
                a=math.radians(azimuth)
                v=np.array([tilt*math.cos(a),tilt*math.sin(a),roll])
                magnitude=np.linalg.norm(v)
                if magnitude>24:continue
                q=([0.,0.,0.,1.] if magnitude==0 else
                   (math.sin(math.radians(magnitude)/2)*v/magnitude).tolist()+[math.cos(math.radians(magnitude)/2)])
                target=anchor@transform({'xyz':[0.,0.,0.],'xyzw':q})
                shift=.12*(target[:3,2]-anchor[:3,2])
                shift[2]=max(-.003,shift[2])
                target[:3,3]+=shift*min(1.,.016/max(1.e-12,np.linalg.norm(shift)))
                yield target


def separated_joint_goals(goals,count):
    """Find a well-spaced subset before choosing an execution order."""
    if not goals:return []
    rotations=np.asarray([t[:3,:3] for t,q in goals])
    cosine=(np.einsum('ijk,ljk->il',rotations,rotations)-1)/2
    distances=np.degrees(np.arccos(np.clip(cosine,-1.,1.)))
    # The capture gate permits two degrees of endpoint error and tests the
    # *measured* poses for 12-degree novelty. Also check distances to recorded
    # observations just before dispatch, since two nominal poses may both drift.
    rng=np.random.RandomState(316)
    best=[];quality=-1.
    for _ in range(360):
        chosen=[]
        for idx in rng.permutation(len(goals)):
            if not chosen or np.min(distances[idx,chosen])>=15.:
                chosen.append(int(idx))
                if len(chosen)==count:break
        separation=(np.min(distances[np.ix_(chosen,chosen)]+180*np.eye(len(chosen))) if len(chosen)>1 else 0.)
        if (len(chosen),separation)>(len(best),quality):best,quality=chosen,separation
    if not best:return []
    # Select contrasting views early enough for the interleaved fit/holdout split.
    ordered=[best.pop(0)]
    while best:
        i=max(best,key=lambda j:min(distances[j,k] for k in ordered))
        best.remove(i);ordered.append(i)
    return [goals[i] for i in ordered]


class CandidateRejected(ValueError):
    pass


def check_sweep(anchor,pose,radius,clearance):
    translation=float(np.linalg.norm(pose[:3,3]-anchor[:3,3]))
    rotation=angle(anchor[:3,:3].T @ pose[:3,:3])
    sweep=translation+2*radius*math.sin(rotation/2)
    # clearance bounds the whole attachment in every direction at the anchor,
    # including downward travel. Compare the worst displacement to that bound.
    if not math.isfinite(sweep) or translation>.02 or rotation>math.radians(25) or sweep+.025>clearance:
        raise CandidateRejected(f'Calibration envelope: shift={translation:.3f} m, turn={math.degrees(rotation):.1f} deg, '
                                f'attachment sweep={sweep:.3f} m, measured clearance={clearance:.3f} m')
    return sweep


def joint_path(start,end):
    distance=max(abs(end[n]-start[n]) for n in start)
    steps=max(1,math.ceil(distance/.005))
    for i in range(1,steps+1):
        yield {name:v+(end[name]-v)*i/steps for name,v in start.items()}


def urdf_pose(origin):
    frame=np.eye(4)
    if origin is None:return frame
    frame[:3,3]=[float(v) for v in origin.get('xyz','0 0 0').split()]
    roll,pitch,yaw=map(float,origin.get('rpy','0 0 0').split())
    sr,sp,sy=map(math.sin,(roll,pitch,yaw))
    cr,cp,cy=map(math.cos,(roll,pitch,yaw))
    frame[:3,:3]=[[cy*cp,cy*sp*sr-sy*cr,cy*sp*cr+sy*sr],
                   [sy*cp,sy*sp*sr+cy*cr,sy*sp*cr-cy*sr],[-sp,cp*sr,cp*cr]]
    return frame


def link_shapes(description,chain):
    """Map collision primitives to FK segments, including side branches on fixed joints."""
    root=ET.fromstring(description)
    links={link.get('name'):link for link in root.findall('link')}
    fixed={}
    for joint in root.findall('joint'):
        child=joint.find('child').get('link')
        if joint.get('type')=='fixed' and child not in chain:
            fixed.setdefault(joint.find('parent').get('link'),[]).append((child,urdf_pose(joint.find('origin'))))
    def geometries(link,offset):
        for collision in links[link].findall('collision'):
            shape=next(iter(collision.find('geometry')))
            frame=offset@urdf_pose(collision.find('origin'))
            if shape.tag=='cylinder':
                dims=[float(shape.get('radius')),float(shape.get('length'))/2]
            elif shape.tag=='sphere':
                dims=[float(shape.get('radius'))]
            elif shape.tag=='box':
                dims=np.array([float(v) for v in shape.get('size').split()])/2
            else:raise ValueError('Unsupported collision geometry for '+link+': '+shape.tag)
            if not np.isfinite(dims).all() or min(dims)<=0:raise ValueError('Invalid URDF collision dimensions for '+link)
            data=(np.array(list(itertools.product((-1,1),repeat=3)))*dims if shape.tag=='box' else dims)
            yield link,frame,shape.tag,data
        for child,frame in fixed.get(link,[]):
            yield from geometries(child,offset@frame)
    return [(index,list(geometries(name,np.eye(4)))) for name,index in chain.items()]


def shape_sweep(old,new,primitive):
    """Upper bound on the displacement of the occupied solid, including its offset origin."""
    _,offset,shape,data=primitive
    start,end=old@offset,new@offset
    delta=end[:3,3]-start[:3,3]
    if shape=='sphere':return float(np.linalg.norm(delta))
    if shape=='box':
        return float(np.max(np.linalg.norm(data@(end[:3,:3]-start[:3,:3]).T+delta,axis=1)))
    # A cylinder is invariant under axial roll. Its support function in a
    # direction making angle t with its axis is r*sin(t)+h*abs(cos(t)),
    # Lipschitz with constant max(r,h) on 0..pi/2. The maximum difference
    # of support functions is the Hausdorff distance of these convex solids.
    cos_tilt=abs(float(start[:3,2]@end[:3,2]))
    tilt=math.acos(float(np.clip(cos_tilt,0.,1.)))
    return float(np.linalg.norm(delta))+max(data)*tilt


def attachment_boxes(path,radius):
    data=json.loads(path.read_text(encoding='utf8'))
    if data.get('frame')!='right_gripper' or data.get('units')!='m' or not data.get('boxes'):
        raise ValueError('Specify installed board and mount boxes in right_gripper metres')
    boxes=[]
    for box in data['boxes']:
        size=np.asarray(box['size_m'],dtype=float)
        if size.shape!=(3,) or not np.isfinite(size).all() or (size<=0).any():
            raise ValueError('Invalid box dimensions')
        offset=transform(box)
        farthest=max(np.linalg.norm(offset[:3,3]+offset[:3,:3]@(size*np.asarray(signs)/2))
                     for signs in itertools.product((-1,1),repeat=3))
        if farthest>radius:raise ValueError('Measured tool radius does not enclose the measured box')
        boxes.append((size.tolist(),packed(offset)))
    return boxes


class Driver:
    topic='/robot/limb/right/joint_command'

    def __init__(self,args):
        import rospy
        import rosgraph
        from baxter_core_msgs.msg import AssemblyState, CollisionAvoidanceState, CollisionDetectionState, EndpointState
        from moveit_msgs.srv import GetPlanningScene, GetPlanningSceneRequest, GetStateValidity
        from moveit_msgs.msg import PlanningSceneComponents, AttachedCollisionObject
        from shape_msgs.msg import SolidPrimitive
        from geometry_msgs.msg import Pose
        from sensor_msgs.msg import JointState
        rospy.init_node('baxter_handeye_local_pose_driver',anonymous=True,disable_signals=True)
        self.ros=rospy; self.master=rosgraph.Master(rospy.get_name())
        self.feedback={}; self.states={}; self.stamps={}; self.pub=None
        self.radius=args.tool_radius_m; self.clearance=args.clearance_m
        self.body_clearance=args.body_clearance_m
        self.joint_names=sorted(joint_names('right'))
        names=joint_names('left') | joint_names('right') | {'head_pan'}
        self.inactive=names.difference(self.joint_names)
        def note(key,m,value):
            now=time.monotonic()
            if hasattr(m,'header'):
                stamp=m.header.stamp.to_sec()
                age=rospy.Time.now().to_sec()-stamp+args.robot_clock_offset_s
                if not -.1<=age<.5 or stamp<=self.stamps.get(key,-math.inf):return
                self.stamps[key]=stamp
                now-=max(age,0.)
            if key in ('collision','avoidance'):self.feedback[key]=(now,value)
            else:self.states[key]=(now,value)
        def joints(m):
            values=dict(zip(m.name,m.position))
            if names.issubset(values) and all(math.isfinite(v) for v in values.values()):
                note('joints',m,values)
        prefix='/robot/limb/right/'
        self.subs=[
            rospy.Subscriber('/robot/joint_states',JointState,joints,queue_size=1),
            rospy.Subscriber('/robot/state',AssemblyState,lambda m:note('enabled',m,m.enabled and not(m.stopped or m.error)),queue_size=1),
            rospy.Subscriber(prefix+'endpoint_state',EndpointState,lambda m:note('endpoint',m,{'position':m.pose.position,'orientation':m.pose.orientation}),queue_size=1),
            rospy.Subscriber(prefix+'collision_avoidance_state',CollisionAvoidanceState,lambda m:note('avoidance',m,{'objects':list(m.collision_object),'other_arm':bool(m.other_arm)}),queue_size=1),
            rospy.Subscriber(prefix+'collision_detection_state',CollisionDetectionState,lambda m:note('collision',m,bool(m.collision_state)),queue_size=1)]
        description=rospy.get_param('/robot_description')
        self.ik=ContinuousIK('right',description)
        chain={self.ik.chain.getSegment(i).getName():i+1 for i in range(self.ik.chain.getNrOfSegments())}
        self.bodies=[(index,shapes) for index,shapes in link_shapes(description,chain) if shapes]
        rospy.wait_for_service('/get_planning_scene',timeout=5)
        rospy.wait_for_service('/check_state_validity',timeout=5)
        query=GetPlanningSceneRequest()
        query.components.components=(PlanningSceneComponents.ROBOT_STATE | PlanningSceneComponents.ROBOT_STATE_ATTACHED_OBJECTS)
        self.scene=rospy.ServiceProxy('/get_planning_scene',GetPlanningScene)(query).scene.robot_state
        if not names.issubset(self.scene.joint_state.name):
            raise ValueError('MoveIt scene is missing robot joints')
        attachment=AttachedCollisionObject();attachment.link_name='right_gripper'
        attachment.object.header.frame_id=attachment.link_name;attachment.object.id='handeye_arm_tool_envelope'
        if args.attachment_boxes:
            for size,pose in attachment_boxes(args.attachment_boxes,self.radius):
                shape=SolidPrimitive(type=SolidPrimitive.BOX,dimensions=size)
                centre=Pose()
                centre.position.x,centre.position.y,centre.position.z=pose['xyz']
                centre.orientation.x,centre.orientation.y,centre.orientation.z,centre.orientation.w=pose['xyzw']
                attachment.object.primitives.append(shape);attachment.object.primitive_poses.append(centre)
        else:
            sphere=SolidPrimitive(type=SolidPrimitive.SPHERE,dimensions=[self.radius])
            centre=Pose();centre.orientation.w=1.
            attachment.object.primitives=[sphere];attachment.object.primitive_poses=[centre]
        # The sphere encloses the hand and the fingers already holding the board.
        # The forearm, torso, head, opposite arm and all world objects stay active.
        attachment.touch_links=['right_gripper','right_gripper_base','right_hand','right_wrist',
                                'r_gripper_l_finger','r_gripper_r_finger','r_gripper_l_finger_tip','r_gripper_r_finger_tip']
        self.scene.attached_collision_objects.append(attachment)
        self.validity=rospy.ServiceProxy('/check_state_validity',GetStateValidity,persistent=True)
        deadline=time.monotonic()+4
        while not {'joints','endpoint','enabled'}.issubset(self.states) or len(self.feedback)<2:
            if time.monotonic()>deadline:raise RuntimeError('Fresh joint, endpoint, enable and collision telemetry required')
            time.sleep(.02)
        # At a stop the encoder can alternate across the model boundary while
        # motionless. Every preview starts from an actual in-limit observation.
        live=self.in_limit_sample(time.monotonic()+3.)
        self.peer={n:live[n] for n in self.inactive}
        self.anchor_joints={n:live[n] for n in self.joint_names}
        self.anchor_bodies=self.link_poses(self.anchor_joints)
        self.anchor=fk_matrix(self.states['endpoint'][1])
        self.check_ownership()
        observed=fk_matrix(self.ik.forward(self.anchor_joints))
        if np.linalg.norm(self.anchor[:3,3]-observed[:3,3])>.005 or angle(self.anchor[:3,:3].T@observed[:3,:3])>.02:
            raise RuntimeError('Measured right_gripper and KDL FK disagree at the anchor')

    def live(self):
        now=time.monotonic()
        if self.ros.is_shutdown() or any(now-self.states[n][0]>=.5 for n in ('joints','endpoint','enabled')):
            raise RuntimeError('ROS stopped or robot telemetry stale')
        if not self.states['enabled'][1]:raise RuntimeError('Robot disabled/stopped/faulted')
        check_controller_feedback(self.feedback,now)
        states=self.states['joints'][1]
        if hasattr(self,'peer') and any(abs(states[n]-self.peer[n])>.012 for n in self.peer):
            raise RuntimeError('Head or peer arm moved during calibration')
        if hasattr(self,'anchor'):
            try:check_sweep(self.anchor,fk_matrix(self.states['endpoint'][1]),self.radius,self.clearance)
            except CandidateRejected as error:raise RuntimeError('Measured endpoint left the local envelope') from error
        return states

    def in_limits(self,joints):
        return all(self.ik.lower[i]<=joints[n]<=self.ik.upper[i] for i,n in enumerate(self.ik.names))

    def in_limit_sample(self,deadline):
        while time.monotonic()<deadline:
            snapshot=self.live()
            if self.in_limits(snapshot):
                return snapshot
            time.sleep(.02)
        raise RuntimeError('No observed joint sample within the URDF limits before the deadline')

    def check_ownership(self):
        pubs=dict(self.master.getSystemState()[0]).get(self.topic,[])
        if set(pubs)-{self.ros.get_name()}:raise RuntimeError('Another right-arm command publisher is registered: '+str(pubs))

    def link_poses(self,joints):
        kdl=self.ik.kdl
        q=kdl.JntArray(len(self.ik.names))
        for i,name in enumerate(self.ik.names):q[i]=joints[name]
        frames=[]
        for index,_ in self.bodies:
            f=kdl.Frame()
            if self.ik.fk.JntToCart(q,f,index)<0:raise RuntimeError('Link forward kinematics unavailable')
            frames.append(transform({'xyz':[f.p[j] for j in range(3)],'xyzw':list(f.M.GetQuaternion())}))
        return frames

    def check(self,joints):
        from moveit_msgs.srv import GetStateValidityRequest
        live=self.live()
        pose=fk_matrix(self.ik.forward(joints))
        check_sweep(self.anchor,pose,self.radius,self.clearance)
        for old,new,(_,shapes) in zip(self.anchor_bodies,self.link_poses(joints),self.bodies):
            for shape in shapes:
                motion=shape_sweep(old,new,shape)
                if not math.isfinite(motion) or motion+.02>self.body_clearance:
                    raise CandidateRejected(f'Link leaves observed free region: {shape[0]}, '
                                            f'occupied-volume sweep {1000*motion:.1f} mm + 20 mm margin '
                                            f'exceeds {1000*self.body_clearance:.1f} mm measured clearance')
        req=GetStateValidityRequest()
        req.robot_state=deepcopy(self.scene)
        js=req.robot_state.joint_state
        js.position=[joints[n] if n in joints else live.get(n,v) for n,v in zip(js.name,js.position)]
        req.robot_state.is_diff=False
        response=self.validity(req)
        if not response.valid:
            raise CandidateRejected('MoveIt collision: '+str([(c.contact_body_1,c.contact_body_2) for c in response.contacts]))

    def recenter(self,start):
        name='right_w1'
        lower=self.ik.lower[self.ik.names.index(name)]
        if not self.in_limits(start) or not 0<=start[name]-lower<=.015:
            raise ValueError('Recenter requires an observed legal right_w1 within 0.015 rad of its lower limit')
        self.check(start)
        reasons=[]
        for shift in (.15,.12,.10,.08,.06,.04):
            goal=dict(start);goal[name]+=shift
            path=list(joint_path(start,goal))
            try:
                for q in path:self.check(q)
            except CandidateRejected as error:
                reasons.append((shift,str(error)));continue
            return path,fk_matrix(self.ik.forward(goal))
        raise CandidateRejected('No inward path passed full preflight: '+str(reasons))

    def cartesian_knots(self,start,target):
        from .primitives import _slerp
        first=fk_matrix(self.ik.forward(start))
        p0,p1=packed(first),packed(target)
        steps=max(1,math.ceil(angle(first[:3,:3].T@target[:3,:3])/math.radians(2)),
                  math.ceil(np.linalg.norm(first[:3,3]-target[:3,3])/.004))
        previous=dict(start)
        for i in range(1,steps+1):
            t=i/steps
            xyz=((1-t)*first[:3,3]+t*target[:3,3]).tolist()
            point=pose_msg(transform({'xyz':xyz,'xyzw':_slerp(p0['xyzw'],p1['xyzw'],t)}))
            solution=self.ik.solve(point,previous,self.ik.forward(previous))
            if max(abs(solution[n]-self.anchor_joints[n]) for n in solution)>.65:
                raise CandidateRejected('Local move would require a large joint reconfiguration')
            yield solution
            previous=solution

    def plan(self,start,target):
        self.check(start)
        path=[];previous=start
        for solution in self.cartesian_knots(start,target):
            for cmd in joint_path(previous,solution):
                self.check(cmd);path.append(cmd)
            previous=solution
        return path

    def joint_route(self,start,goal):
        self.check(start)
        def segment(a,b):
            points=list(joint_path(a,b))
            for cmd in points:self.check(cmd)
            return points
        try:return segment(start,goal)
        except CandidateRejected as direct_error:
            if start==self.anchor_joints or goal==self.anchor_joints:raise
            # A straight connection across two admissible wrist poses can cross
            # an obstacle. Try the calibrated start with both legs fully checked.
            try:
                path=segment(start,self.anchor_joints)+segment(self.anchor_joints,goal)
            except CandidateRejected as anchor_error:
                raise CandidateRejected(f'Direct wrist segment rejected ({direct_error}); '
                                        f'via anchor rejected ({anchor_error})') from anchor_error
            self.record('via_anchor',reason=str(direct_error),command_points=len(path))
            return path

    def lattice_route(self,count,record):
        from collections import Counter
        goals=[]; rejected=Counter()
        for target in wrist_lattice(self.anchor):
            try:
                check_sweep(self.anchor,target,self.radius,self.clearance)
                solution=None
                for solution in self.cartesian_knots(self.anchor_joints,target):pass
                self.check(solution)
            except (CandidateRejected,IKUnreachableError) as error:
                rejected[str(error).split(',')[0]]+=1
                continue
            goals.append((fk_matrix(self.ik.forward(solution)),solution))
        selected=separated_joint_goals(goals,count)
        record('wrist_lattice',feasible_endpoints=len(goals),distinct=len(selected),rejections=dict(rejected))
        if len(selected)<count:raise RuntimeError(f'Only {len(selected)} of {count} distinct wrist views passed preflight')
        current=self.anchor_joints
        for i,(target,goal) in enumerate(selected):
            path=self.joint_route(current,goal)
            record('preflight',index=i,pose=packed(target),joints=goal,command_points=len(path))
            current=goal
        return selected

    def enable_publisher(self):
        from baxter_core_msgs.msg import JointCommand
        self.check_ownership()
        self.pub=self.ros.Publisher(self.topic,JointCommand,queue_size=1)
        deadline=time.monotonic()+3
        while not self.pub.get_num_connections():
            if time.monotonic()>=deadline:raise RuntimeError('No Baxter joint-command subscriber')
            self.live();time.sleep(.05)

    def stop_at_feedback(self):
        from baxter_core_msgs.msg import JointCommand
        if self.pub is None:return
        self.check_ownership()
        sample=self.states['joints']
        if time.monotonic()-sample[0]>.3 or not self.pub.get_num_connections():
            raise RuntimeError('Cannot confirm sufficiently recent feedback and command connection for hold')
        if not self.in_limits(sample[1]):raise RuntimeError('Hold sample outside model limits; no new setpoint issued')
        self.pub.publish(JointCommand(mode=JointCommand.POSITION_MODE,names=self.joint_names,
                                     command=[sample[1][name] for name in self.joint_names]))
        time.sleep(.1)

    def command_sample(self, inward_only):
        """Only the inward-only retreat may wait out an encoder tick across w1's lower stop."""
        deadline = time.monotonic() + .5
        while True:
            sample = self.live()
            if self.in_limits(sample):
                return sample
            bad = [(name, self.ik.lower[i] - sample[name]) for i, name in enumerate(self.ik.names)
                   if not self.ik.lower[i] <= sample[name] <= self.ik.upper[i]]
            if (not inward_only or len(bad) != 1 or bad[0][0] != 'right_w1'
                    or not 0 < bad[0][1] <= .002 or time.monotonic() >= deadline):
                raise RuntimeError('Measured joints outside URDF limits before command: '+str(bad))
            self.check_ownership()
            time.sleep(.01)

    def checked_command_start(self, previous, goal, inward_only):
        # Baxter can still be settling to its last checked command while MoveIt
        # answers. Recheck the actual path from the newer start before publishing.
        deadline = time.monotonic() + .7
        while True:
            self.check_ownership()
            live = self.command_sample(inward_only)
            if max(abs(live[n]-previous[n]) for n in goal) > .012:
                raise RuntimeError('Start no longer matches checked path')
            for q in joint_path({n:live[n] for n in goal},goal):
                self.check(q)
            self.check_ownership()
            latest = self.command_sample(inward_only)
            if max(abs(latest[n]-live[n]) for n in goal) <= .003:
                return latest
            if time.monotonic() >= deadline:
                raise RuntimeError('Arm did not settle during segment revalidation')

    def execute(self,path,start,target,inward_only=False):
        from baxter_core_msgs.msg import JointCommand
        previous=start
        for goal in path:
            latest=self.checked_command_start(previous,goal,inward_only)
            if inward_only and latest['right_w1']>goal['right_w1']:
                if latest['right_w1']-goal['right_w1']>.005:raise RuntimeError('Unexpected inward overshoot before command')
                previous=goal;continue
            dt=max(abs(goal[n]-previous[n]) for n in goal)/.05
            self.pub.publish(JointCommand(mode=JointCommand.POSITION_MODE,names=self.joint_names,command=[goal[n] for n in self.joint_names]))
            self.record('joint_command_sent',goal=goal,start={n:latest[n] for n in goal})
            began=time.monotonic()
            while True:
                measured=self.live()
                error=max(abs(measured[n]-goal[n]) for n in goal)
                elapsed=time.monotonic()-began
                if error>.035 or elapsed>2.:raise RuntimeError('Stopped on tracking error or joint following timeout')
                if elapsed>=max(dt,.03) and error<.01:break
                time.sleep(.01)
            previous=goal
        # Joint feedback and endpoint feedback arrive on independent ROS topics.
        # Require fresh endpoint measurements after the final joint has settled.
        completed=time.monotonic()
        previous_stamp=completed
        stable_since=None
        stable_pose=None
        while time.monotonic()-completed<1.0:
            self.check_ownership()
            self.live()
            stamp,measurement=self.states['endpoint']
            if stamp>previous_stamp:
                previous_stamp=stamp
                end=fk_matrix(measurement)
                inside=(np.linalg.norm(end[:3,3]-target[:3,3])<=.005
                        and angle(end[:3,:3].T@target[:3,:3])<=math.radians(2))
                if not inside:
                    stable_since=None
                elif (stable_since is None or np.linalg.norm(end[:3,3]-stable_pose[:3,3])>.0015
                      or angle(end[:3,:3].T@stable_pose[:3,:3])>math.radians(.75)):
                    stable_since,stable_pose=stamp,end
                elif stamp-stable_since>=.08:
                    return
            time.sleep(.01)
        raise RuntimeError('Fresh stable endpoint feedback did not reach the calibration target')


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--tool-radius-m',type=float,required=True,help='Measured radius about right_gripper enclosing the entire installed board, mount and fingers')
    p.add_argument('--clearance-m',type=float,required=True,help='Measured lower bound on initial clearance of the entire attachment to the table and surrounding obstacles')
    p.add_argument('--body-clearance-m',type=float,required=True,help='Smallest verified clearance from moving right arm link bodies to the table and other external obstacles at the anchor')
    p.add_argument('--attachment-boxes',type=Path,help='Measured oriented boxes covering board/mount in right_gripper coordinates. Defaults to enclosing sphere.')
    p.add_argument('--count',type=int,default=16)
    p.add_argument('--wrist-lattice',action='store_true',help='Use small secondary tilts with wrist rolls and FK-checked joint interpolation')
    p.add_argument('--robot-clock-offset-s',type=float,default=0.)
    p.add_argument('--drive',action='store_true')
    p.add_argument('--recenter-w1',action='store_true',help='Check a single inward-only move when the right wrist is at its lower model limit; no capture')
    p.add_argument('--capture-dir',type=Path,help='Fresh handeye_capture directory, using --pose-gate DIR/request.json')
    p.add_argument('--output',type=Path,required=True,help='New append-only JSONL audit path')
    args=p.parse_args(argv)
    if (not .03<=args.tool_radius_m<=.8 or not .03<=args.clearance_m<=2
            or not .03<=args.body_clearance_m<=2 or not -.5<=args.robot_clock_offset_s<=.5 or not 1<=args.count<=19):
        p.error('Invalid attachment radius, clearance, clock offset or sample count')
    if args.drive and not args.recenter_w1 and not args.capture_dir:p.error('--drive requires a fresh --capture-dir')
    if args.recenter_w1 and args.capture_dir:p.error('Recenter and gated capture are separate operations')
    if args.wrist_lattice and args.recenter_w1:p.error('Use wrist lattice and recenter separately')
    with args.output.open('x',encoding='utf8',buffering=1) as audit:
        def record(event,**kw):audit.write(json.dumps({'event':event,'time':time.time(),**kw},allow_nan=False)+'\n')
        driver=Driver(args)
        driver.record=record
        poses=[];fixed_goals=[];current=driver.anchor_joints
        anchor=packed(driver.anchor)
        record('anchor',pose=anchor,joints=current,tool_radius_m=args.tool_radius_m,
               clearance_m=args.clearance_m,body_clearance_m=args.body_clearance_m)
        driver.check(current)
        if args.recenter_w1:
            path,target=driver.recenter(current)
            record('recenter_preflight',start=current,goal=path[-1],pose=packed(target),command_points=len(path))
        elif args.wrist_lattice:
            for target,goal in driver.lattice_route(args.count,record):
                poses.append(target);fixed_goals.append(goal)
        else:
            for i,target in enumerate(candidate_poses(driver.anchor)):
                if not distinct_pose(target,poses):continue
                try:path=driver.plan(current,target)
                except (CandidateRejected,IKUnreachableError) as error:
                    record('candidate_rejected',candidate=i,reason=str(error))
                    continue
                current=path[-1];poses.append(target)
                record('preflight',index=len(poses)-1,candidate=i,pose=packed(target),command_points=len(path))
                if len(poses)==args.count:break
            if len(poses)<args.count:raise RuntimeError(f'Only {len(poses)} of {args.count} stationary poses passed preflight')
        if not args.drive:return 0
        import fcntl
        import os
        import tempfile
        cap=args.capture_dir; gate=cap/'request.json' if cap else None
        def captured():
            return [entry for name in ('train','holdout') for entry in json.loads((cap/(name+'.json')).read_text())]
        if cap and (gate.exists() or captured()):raise RuntimeError('Start a fresh gated capture before driving')
        with open(Path(tempfile.gettempdir())/'baxter-right-handeye-drive.lock','a') as lock:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                now=driver.live()
                if any(abs(now[n]-driver.anchor_joints[n])>.008 for n in driver.anchor_joints):
                    raise RuntimeError('Arm changed during preview')
                driver.enable_publisher()
                if args.recenter_w1:
                    actual=driver.in_limit_sample(time.monotonic()+3.)
                    measured={n:actual[n] for n in driver.joint_names}
                    path,target=driver.recenter(measured)
                    record('recenter_dispatch',start=measured,goal=path[-1])
                    driver.execute(path,measured,target,inward_only=True)
                    record('recenter_completed',joints=driver.live())
                    return 0
                for i,target in enumerate(poses):
                    live=driver.live()
                    measured={n:live[n] for n in driver.joint_names}
                    if fixed_goals:
                        measured_priors=[transform(s['base_T_effector']) for s in captured()]
                        closest=min((math.degrees(angle(t[:3,:3].T@target[:3,:3])) for t in measured_priors),default=180.)
                        if closest<14.1:
                            raise RuntimeError(f'Wrist view {i} too close to recorded posture ({closest:.2f} deg)')
                    path=driver.joint_route(measured,fixed_goals[i]) if fixed_goals else driver.plan(measured,target)
                    record('moving',index=i)
                    driver.execute(path,measured,target)
                    request_id=uuid.uuid4().hex
                    req={'id':request_id,'base_T_effector':packed(target)}
                    tmp=gate.with_suffix('.tmp')
                    tmp.write_text(json.dumps(req),encoding='utf8');os.replace(str(tmp),str(gate))
                    try:
                        deadline=time.monotonic()+15
                        while not any(s.get('drive_pose_id')==request_id for s in captured()):
                            driver.live();driver.check_ownership()
                            if time.monotonic()>deadline:raise RuntimeError('Capture did not accept this stationary pose')
                            time.sleep(.12)
                    finally:
                        if gate.exists():gate.unlink()
                    record('captured',index=i,id=request_id)
            except BaseException as error:
                try:
                    driver.stop_at_feedback()
                    record('stopped',reason=str(error),hold_requested=driver.pub is not None)
                except Exception as hold_error:
                    record('stopped',reason=str(error),hold_requested=False,hold_error=str(hold_error))
                raise
            finally:
                if driver.pub:driver.pub.unregister()
    return 0


if __name__=='__main__':raise SystemExit(main())
