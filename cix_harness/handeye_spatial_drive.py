"""Plan and optionally collect a preregistered right-arm hand-eye volume on ROS 1."""
import argparse
from copy import copy, deepcopy
import hashlib
import itertools
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

import numpy as np

from .handeye import angle, packed, transform
from .handeye_capture import write_json
from .handeye_drive import (CandidateRejected, Driver, attachment_boxes, fk_matrix, joint_path,
                            link_shapes, pose_msg, shape_sweep, urdf_pose)
from .handeye_volume import validate_plan
from .kinematics import IKUnreachableError
from .moveit_probe import pose_request
from .primitives import _slerp
from .trajectory import check_controller_feedback


# The existing driver stops when position tracking error exceeds .035 rad.
# Plan the endpoint farther inside than this bound, allowing encoder quantization.
TARGET_JOINT_MARGIN_RAD = .04


def guarded_joint_step(previous, state, lower, upper, *, final=False):
    """Only an actual legal near-limit start may use the inside exit corridor."""
    for name in state:
        before, after = previous[name], state[name]
        lo,hi=lower[name],upper[name]
        if not math.isfinite(after) or not lo<=before<=hi or not lo<=after<=hi:
            raise CandidateRejected('Actual or proposed joint outside URDF: '+name)
        inward_lo, inward_hi = lo+TARGET_JOINT_MARGIN_RAD, hi-TARGET_JOINT_MARGIN_RAD
        if ((after < inward_lo-1e-8 and (final or after < before-1e-8))
                or (after > inward_hi+1e-8 and (final or after > before+1e-8))):
            raise CandidateRejected(f'{name} approaches its URDF limit or ends within {TARGET_JOINT_MARGIN_RAD:.3f} rad')


def box_bounds(row):
    lo, hi = (np.asarray(row[key], dtype=float) for key in ('lower', 'upper'))
    if lo.shape != (3,) or hi.shape != (3,) or not np.isfinite([lo, hi]).all() or np.any(hi <= lo):
        raise ValueError('Expected ordered finite bounds in base metres')
    return lo, hi


def validate_scene(scene):
    if scene.get('frame') != 'base' or scene.get('units') != 'm' or not scene.get('source'):
        raise ValueError('Survey must name its source, base frame and metre units')
    ids = set()
    for obj in scene['world_boxes']:
        if not obj['id'] or obj['id'] in ids:
            raise ValueError('Collision box IDs must be nonempty and unique')
        ids.add(obj['id'])
        box_bounds(obj)
    if not ids:
        raise ValueError('Supply the measured tabletop and observed obstacles')
    for region in scene['verified_free_sweep_boxes']:
        if not region.get('source'):
            raise ValueError('Each verified free region needs measurement provenance')
        box_bounds(region)
    for region in scene.get('verified_scene_coverage_boxes', []):
        if (not isinstance(region.get('source'),str) or not region['source'].strip()
                or region.get('contents')!='robot_and_surveyed_world_only'):
            raise ValueError('Scene-coverage region needs physical evidence of its modeled contents')
        box_bounds(region)
    anchor = scene['anchor']
    for key in ('body_clearance_m', 'tool_clearance_m'):
        if not math.isfinite(anchor[key]) or anchor[key] < 0:
            raise ValueError('Anchor clearances must be measured nonnegative lower bounds')
    if not .03 <= scene['tool_radius_m'] <= .8:
        raise ValueError('Invalid supplementary tool envelope radius')
    for pose in (scene['base_T_camera'], scene['effector_T_tag'], anchor['base_T_effector']):
        transform(pose)


def marker_projection(goal, scene):
    camera_T_tag = np.linalg.inv(transform(scene['base_T_camera'])) @ goal @ transform(scene['effector_T_tag'])
    intr = scene['intrinsics']
    half = scene.get('tag_size_m', .10) / 2
    corners = np.array([[-half, -half, 0], [half, -half, 0], [half, half, 0], [-half, half, 0]])
    corners = corners @ camera_T_tag[:3, :3].T + camera_T_tag[:3, 3]
    if np.any(corners[:, 2] <= 0):
        raise CandidateRejected('Marker behind the color camera')
    pixels = corners[:, :2] / corners[:, 2:3] * [intr['fx'], intr['fy']] + [intr['cx'], intr['cy']]
    z = camera_T_tag[:3, 2]
    ray = camera_T_tag[:3, 3]
    ref = np.linalg.inv(transform(scene['base_T_camera'])) @ transform(scene['anchor']['base_T_effector']) @ transform(scene['effector_T_tag'])
    cosine = np.sign(ref[:3, 2] @ ref[:3, 3]) * (z @ ray) / np.linalg.norm(ray)
    if (cosine < .30 or np.any(pixels < 8)
            or np.any(pixels >= np.asarray([intr['width'], intr['height']])-8)):
        raise CandidateRejected('Full marker is outside predicted field of view or viewed edge-on')
    return {'corners_px': pixels.tolist(), 'front_cosine': float(cosine), 'occlusion_verified': False}


def bounds_of_shape(frame, primitive):
    _, offset, kind, data = primitive
    pose = frame @ offset
    axis = pose[:3, 2]
    if kind == 'sphere':
        half = np.full(3, data[0])
    elif kind == 'cylinder':
        half = np.abs(axis) * data[1] + np.sqrt(np.maximum(0, 1-axis**2)) * data[0]
    else:
        half = np.max(np.abs(data @ pose[:3, :3].T), axis=0)
    return pose[:3, 3] - half, pose[:3, 3] + half


def covered(lo, hi, boxes):
    """Whether the entire query box is inside the exact union of surveyed boxes."""
    if any(np.all(lo >= a) and np.all(hi <= b) for a,b in boxes):
        return True
    remaining=[(lo,hi)]
    for a,b in boxes:
        following=[]
        for lower,upper in remaining:
            cut_lo,cut_hi=np.maximum(lower,a),np.minimum(upper,b)
            if np.any(cut_hi<=cut_lo):
                following.append((lower,upper))
                continue
            # Subtract this intersection as up to six disjoint rectangular slabs.
            lower,upper=lower.copy(),upper.copy()
            for axis in range(3):
                if lower[axis] < cut_lo[axis]:
                    cap=upper.copy();cap[axis]=cut_lo[axis]
                    following.append((lower.copy(),cap));lower[axis]=cut_lo[axis]
                if upper[axis] > cut_hi[axis]:
                    cap=lower.copy();cap[axis]=cut_hi[axis]
                    following.append((cap,upper.copy()));upper[axis]=cut_hi[axis]
        if not following:
            return True
        remaining=following
    return False


def covers_primitive(frame, primitive, pad, contains):
    """Prove the padded URDF solid lies inside the surveyed union, including at its corners."""
    lo,hi=bounds_of_shape(frame,primitive)
    if contains(lo-pad,hi+pad):
        return True
    _,offset,kind,data=primitive
    pose=frame@offset
    rot,centre=pose[:3,:3],pose[:3,3]
    if kind=='box':
        half=np.max(np.abs(data),axis=0)
    else:
        radius=data[0]
        half=np.array([radius,radius,data[1] if kind=='cylinder' else radius])
    pending=[(-half,half)]
    while pending:
        local_lo,local_hi=pending.pop()
        origin_distance=np.maximum(np.maximum(local_lo,-local_hi),0.)
        if kind in ('sphere','cylinder'):
            d=origin_distance if kind=='sphere' else origin_distance[:2]
            if d@d > radius**2+1e-14:
                continue
        mid=(local_lo+local_hi)*.5
        half_size=(local_hi-local_lo)*.5
        world_center=centre+rot@mid
        world_half=abs(rot)@half_size+pad
        if contains(world_center-world_half,world_center+world_half):
            continue
        axis=int(np.argmax(half_size))
        if 2*half_size[axis] <= .012:
            return False
        left_hi,right_lo=local_hi.copy(),local_lo.copy()
        left_hi[axis]=right_lo[axis]=mid[axis]
        pending.extend([(local_lo,left_hi),(right_lo,local_hi)])
    return True


def nearest_order(start, poses, tool_radius):
    """Preserve registered targets and splits; reduce unnecessary transfer travel."""
    remaining = list(poses)
    ordered = []
    while remaining:
        def cost(row):
            target = transform(row['base_T_effector'])
            return (np.linalg.norm(target[:3,3]-start[:3,3])
                    + 2*tool_radius*math.sin(angle(start[:3,:3].T@target[:3,:3])/2), row['id'])
        row = min(remaining, key=cost)
        remaining.remove(row)
        ordered.append(row)
        start = transform(row['base_T_effector'])
    return ordered


def box_shape(name, frame, size):
    corners = np.array(list(itertools.product((-1.,1.),repeat=3))) * (np.asarray(size)/2)
    return name,frame,'box',corners


def native_finger_envelopes(description):
    """Tool-frame bounds of the native fingers over their full prismatic travel."""
    root=ET.fromstring(description)
    joints={j.find('child').get('link'):j for j in root.findall('joint')}
    eef=joints['right_gripper']
    if eef.get('type')!='fixed':
        raise ValueError('Expect a fixed right_gripper endpoint')
    base=eef.find('parent').get('link')
    eef_T_base=np.linalg.inv(urdf_pose(eef.find('origin')))
    result=[]
    for finger in ('r_gripper_l_finger','r_gripper_r_finger'):
        slider=joints[finger]
        if slider.get('type')!='prismatic' or slider.find('parent').get('link')!=base:
            raise ValueError('Unexpected native finger model')
        axis=np.fromstring(slider.find('axis').get('xyz'),sep=' ')
        if abs(np.linalg.norm(axis)-1)>.00001:
            raise ValueError('Invalid URDF finger slide axis')
        ends=[]
        for limit in ('lower','upper'):
            offset=np.eye(4);offset[:3,3]=axis*float(slider.find('limit').get(limit))
            ends.append(eef_T_base@urdf_pose(slider.find('origin'))@offset)
        for primitive in link_shapes(description,{finger:1})[0][1]:
            limits=[bounds_of_shape(E,primitive) for E in ends]
            lo=np.minimum(limits[0][0],limits[1][0]);hi=np.maximum(limits[0][1],limits[1][1])
            box=np.eye(4);box[:3,3]=(lo+hi)/2
            result.append(box_shape(primitive[0],box,hi-lo))
    if not result:
        raise ValueError('Missing native finger collision shapes')
    return result


def collision_box(row):
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import CollisionObject
    from shape_msgs.msg import SolidPrimitive
    lo, hi = box_bounds(row)
    obj = CollisionObject()
    obj.id = row['id']
    obj.header.frame_id = 'base'
    obj.primitives.append(SolidPrimitive(type=SolidPrimitive.BOX, dimensions=(hi-lo).tolist()))
    centre = Pose()
    centre.orientation.w = 1.
    centre.position.x, centre.position.y, centre.position.z = .5*(hi+lo)
    obj.primitive_poses.append(centre)
    return obj


class SpatialDriver(Driver):
    def __init__(self, args, survey):
        self.survey = survey
        self.free_boxes = [box_bounds(b) for b in survey['verified_free_sweep_boxes']]
        self.scene_boxes = [box_bounds(b) for b in survey.get('verified_scene_coverage_boxes', [])]
        self.gaps = set()
        self.gap_bounds = {}
        self.driving = False
        self.last_measured_check = None
        super().__init__(args)
        import actionlib
        from moveit_msgs.msg import MoveGroupAction
        self.move_group = actionlib.SimpleActionClient('/move_group', MoveGroupAction)
        if not self.move_group.wait_for_server(self.ros.Duration(5)):
            raise RuntimeError('Plan-only MoveGroup action unavailable')
        self.world = [collision_box(row) for row in survey['world_boxes']]
        if args.attachment_boxes:
            self.tool_shapes=native_finger_envelopes(self.ros.get_param('/robot_description'))
            self.tool_shapes.extend(box_shape('installed_board',transform(pose),size)
                                    for size,pose in attachment_boxes(args.attachment_boxes,self.radius))
        else:
            self.tool_shapes=[('installed_tool',np.eye(4),'sphere',[self.radius])]
        # Keep the recorded anchor even when resuming from a later sample.
        recorded = survey['anchor']
        if set(recorded['joints']) != set(self.joint_names):
            raise ValueError('Coverage model requires all surveyed right arm joint angles')
        self.ref_bodies = self.link_poses(recorded['joints'])
        self.ref_pose = transform(recorded['base_T_effector'])
        predicted = fk_matrix(self.ik.forward(recorded['joints']))
        if (np.linalg.norm(predicted[:3, 3] - self.ref_pose[:3, 3]) > .005
                or angle(predicted[:3, :3].T @ self.ref_pose[:3, :3]) > .035):
            raise ValueError('Survey reference and FK do not agree')
        self._sample_scene = deepcopy(self.scene)
        # For the read-only state-validity service, add the same boxes fixed to
        # base in the request only. No publication to the global planning scene.
        from moveit_msgs.msg import AttachedCollisionObject
        from moveit_msgs.srv import GetStateValidityRequest
        for obj in self.world:
            item = AttachedCollisionObject()
            item.link_name = 'base'
            item.object = deepcopy(obj)
            item.touch_links = ['base']
            self._sample_scene.attached_collision_objects.append(item)
        # The main planning/execution loop is serialized. Keep static meshes and
        # boxes once and update only the measured/target joints for each query.
        self._state_request = GetStateValidityRequest()
        self._state_request.robot_state = self._sample_scene
        self._state_request.robot_state.is_diff = False
        self._passive = list(self._sample_scene.joint_state.position)

    def live(self):
        now = time.monotonic()
        if self.ros.is_shutdown() or any(now-self.states[k][0] >= .5 for k in ('joints','endpoint','enabled')):
            raise RuntimeError('Current robot telemetry unavailable')
        if not self.states['enabled'][1]:
            raise RuntimeError('Robot disabled or stopped')
        check_controller_feedback(self.feedback, now)
        live = self.states['joints'][1]
        if hasattr(self, 'peer') and any(abs(live[n]-self.peer[n]) > .012 for n in self.peer):
            raise RuntimeError('Head or inactive arm moved during the spatial survey')
        if self.driving and self.states['joints'][0] != self.last_measured_check:
            self._check_state({n:live[n] for n in self.joint_names}, live)
            self.last_measured_check = self.states['joints'][0]
        return live

    def check(self, joints):
        self._check_state(joints, self.live())

    def missing_volume(self, link, lo, hi):
        self.gaps.add(link)
        prior = self.gap_bounds.get(link, (lo,hi))
        self.gap_bounds[link] = (np.minimum(prior[0],lo),np.maximum(prior[1],hi))

    def coverage_report(self):
        return {name:dict(lower=lo.tolist(),upper=hi.tolist()) for name,(lo,hi) in self.gap_bounds.items()}

    def _check_state(self, joints, live):
        if not self.in_limits(joints):
            raise CandidateRejected('Joint state exceeds URDF limits')
        pose = fk_matrix(self.ik.forward(joints))
        before = self.ref_pose
        centre = pose[:3, 3]
        shift = np.linalg.norm(centre-before[:3, 3])
        turn = angle(before[:3, :3].T @ pose[:3, :3])
        swept = shift + 2*self.radius*math.sin(turn/2)
        gaps = set()
        if swept+.025 > self.clearance:
            for shape in self.tool_shapes:
                if shape_sweep(before,pose,shape)+.025 <= self.clearance:
                    continue
                lo,hi=bounds_of_shape(pose,shape)
                if not covers_primitive(pose,shape,.025,self.coverage_contains):
                    gaps.add(shape[0]);self.missing_volume(shape[0],lo-.025,hi+.025)
        for old, new, (_, shapes) in zip(self.ref_bodies, self.link_poses(joints), self.bodies):
            for primitive in shapes:
                if shape_sweep(old, new, primitive)+.020 <= self.body_clearance:
                    continue
                lo, hi = bounds_of_shape(new, primitive)
                if not covers_primitive(new,primitive,.020,self.coverage_contains):
                    gaps.add(primitive[0])
                    self.missing_volume(primitive[0],lo-.020,hi+.020)
        if gaps and (self.driving or self.scene_boxes):
            raise CandidateRejected('Missing measured obstacle coverage: '+', '.join(sorted(gaps)))
        js = self._state_request.robot_state.joint_state
        js.position = [joints[n] if n in joints else live.get(n, v) for n, v in zip(js.name, self._passive)]
        response = self.validity(self._state_request)
        if not response.valid:
            pairs = [(c.contact_body_1, c.contact_body_2) for c in response.contacts]
            raise CandidateRejected('MoveIt full-state collision: '+str(pairs))

    def coverage_contains(self,lo,hi):
        # A measured empty region can contain the future moving body. A
        # separately surveyed region may also contain modeled obstacles; every
        # state must still pass the scene collision query below.
        return covered(lo,hi,self.free_boxes+self.scene_boxes)

    def interior_ik(self):
        if not hasattr(self,'_interior_ik'):
            ik=copy(self.ik)
            k=ik.kdl
            lo,hi=k.JntArray(7),k.JntArray(7)
            # A small numerical reserve keeps solutions on a soft IK limit from
            # falling outside the independently checked command margin.
            for i in range(7):
                lo[i]=ik.lower[i]+TARGET_JOINT_MARGIN_RAD+.001
                hi[i]=ik.upper[i]-TARGET_JOINT_MARGIN_RAD-.001
            ik.solver=k.ChainIkSolverPos_NR_JL(ik.chain,lo,hi,ik.fk,ik.velocity,150,1e-6)
            self._interior_ik=ik
        return self._interior_ik

    def checked_command_start(self,previous,goal,inward_only):
        measured=super().checked_command_start(previous,goal,inward_only)
        lower={n:self.ik.lower[i] for i,n in enumerate(self.ik.names)}
        upper={n:self.ik.upper[i] for i,n in enumerate(self.ik.names)}
        guarded_joint_step(measured,goal,lower,upper)
        return measured

    def execute(self,path,start,target,inward_only=False):
        lower={n:self.ik.lower[i] for i,n in enumerate(self.ik.names)}
        upper={n:self.ik.upper[i] for i,n in enumerate(self.ik.names)}
        def pending():
            for i,goal in enumerate(path):
                measured=self.live()
                try:
                    guarded_joint_step(measured,goal,lower,upper,final=i==len(path)-1)
                except CandidateRejected:
                    # A legal, measured pose may already have passed an interim
                    # inward waypoint. Never publish an outward correction.
                    if (i==len(path)-1 or not self.in_limits(measured)
                            or max(abs(measured[n]-goal[n]) for n in goal)>.01):
                        raise
                    self.record('passed_inward_waypoint',sample=i)
                    continue
                yield goal
        super().execute(pending(),start,target,inward_only)

    def near_knots(self,start,target):
        first=fk_matrix(self.ik.forward(start))
        q0,q1=packed(first)['xyzw'],packed(target)['xyzw']
        steps=max(1,math.ceil(np.linalg.norm(first[:3,3]-target[:3,3])/.006),
                  math.ceil(angle(first[:3,:3].T@target[:3,:3])/.035))
        previous=start
        knots=[]
        for i in range(1,steps+1):
            f=i/steps
            pose=transform({'xyz':((1-f)*first[:3,3]+f*target[:3,3]).tolist(),'xyzw':_slerp(q0,q1,f)})
            previous=self.interior_ik().solve(pose_msg(pose),previous,self.ik.forward(previous))
            knots.append(previous)
        return knots

    def endpoint_branches(self,start,target):
        """Find deterministic alternatives for the redundant elbow with bounded KDL."""
        ik=self.interior_ik()
        goal_msg=pose_msg(target).pose
        goal=ik.frame(goal_msg.position,goal_msg.orientation)
        candidates=[]
        trial_seeds=[]
        for offset in (0.,-.25,.25,-.55,.55,-.85,.85,-1.15,1.15):
            for shoulder in (0.,-.35,.35):
                bias={'right_e0':offset,'right_s0':shoulder,'right_w0':-.5*offset}
                trial_seeds.append([max(ik.lower[i],min(ik.upper[i],start[n]+bias.get(n,0.))) for i,n in enumerate(ik.names)])
        # Seeding only the current elbow can miss a disconnected IK family.
        # Cover the bounded 7-D seed box deterministically, with a Halton grid.
        def radical_inverse(index,base):
            value=0.;place=1/base
            while index:
                index,digit=divmod(index,base);value+=digit*place;place/=base
            return value
        for trial in range(1,129):
            trial_seeds.append([ik.lower[i]+(ik.upper[i]-ik.lower[i])*radical_inverse(trial,base)
                                for i,base in enumerate((2,3,5,7,11,13,17))])
        for values in trial_seeds:
            seed=ik.kdl.JntArray(7);out=ik.kdl.JntArray(7)
            for i,v in enumerate(values):seed[i]=v
            if ik.solver.CartToJnt(seed,goal,out)<0:continue
            q={n:float(out[i]) for i,n in enumerate(ik.names)}
            if not self.in_limits(q) or not all(map(math.isfinite,q.values())):continue
            if any(max(abs(q[n]-other[n]) for n in q)<.035 for other in candidates):continue
            reached=fk_matrix(ik.forward(q))
            if np.linalg.norm(reached[:3,3]-target[:3,3])>.001 or angle(reached[:3,:3].T@target[:3,:3])>.005:
                continue
            candidates.append(q)
        return sorted(candidates,key=lambda q:sum((q[n]-start[n])**2 for n in q))

    def planned_path(self,start,target):
        # A previous plan can preserve the redundant elbow branch. Rebuild the
        # full path from the current joints and validate each state afresh.
        for hint in self.survey.get('joint_path_hints', ()):
            if not np.allclose(transform(hint['base_T_effector']),target,atol=1e-10,rtol=0):
                continue
            knots=hint['joint_knots']
            if (not isinstance(knots,list) or not knots
                    or any(set(q)!=set(start) or not all(map(math.isfinite,q.values())) for q in knots)):
                raise ValueError('Malformed joint path hint')
            try:
                path=self.checked_knots(start,target,knots,reject_detour=True)
                self.last_route='revalidated_joint_hint'
                return path
            except CandidateRejected as error:
                self.record('joint_path_hint_rejected',reason=str(error))
        near_goal=None
        try:
            knots=self.near_knots(start,target)
            near_goal=knots[-1]
            path=self.checked_knots(start,target,knots)
            self.last_route='local_cartesian_kdl'
            return path
        except (IKUnreachableError,CandidateRejected) as error:
            self.record('local_path_rejected',reason=str(error))
        if near_goal is not None:
            try:
                path=self.checked_knots(start,target,[near_goal],reject_detour=True)
                self.last_route='direct_joint_near_goal'
                return path
            except CandidateRejected as error:
                self.record('direct_joint_rejected',reason=str(error))
        seeds=self.endpoint_branches(start,target)
        def alternatives():
            yield from seeds
            from .handeye_elbow import redundant_branches
            pose=pose_msg(target).pose
            yield from redundant_branches(self.interior_ik(),self.ik.frame(pose.position,pose.orientation),
                                           start,seeds,TARGET_JOINT_MARGIN_RAD)
        for branch in alternatives():
            try:
                self.check(branch)
                path=self.checked_knots(start,target,[branch],reject_detour=True)
                self.last_route='checked_elbow_branch'
                return path
            except CandidateRejected:
                continue
        last_error=None
        for goal in ([near_goal,None] if near_goal is not None else [None]):
            try:
                return self.moveit_path(start,target,goal)
            except CandidateRejected as error:
                last_error=error
                self.record('moveit_candidate_rejected',near_joint_goal=goal is not None,reason=str(error))
        raise last_error

    def moveit_path(self, start, target, joint_goal=None):
        import genpy.message
        from moveit_msgs.msg import MoveGroupGoal
        position = packed(target)
        request = pose_request('right', 'right_arm', position['xyz'], position['xyzw'])
        constraints = request['goal_constraints'][0]
        constraints['position_constraints'][0]['constraint_region']['primitives'][0]['dimensions'] = [.0025]
        for axis in 'xyz':
            constraints['orientation_constraints'][0]['absolute_'+axis+'_axis_tolerance'] = .012
        if joint_goal is not None:
            request['goal_constraints']=[{'joint_constraints':[
                {'joint_name':n,'position':v,'tolerance_above':.001,'tolerance_below':.001,'weight':1.}
                for n,v in joint_goal.items()]}]
        # OMPL must honor joint limits on the entire generated path as well.
        # A start already inside the edge band is constrained to its measured
        # side; every returned sample also gets the monotonic exit check.
        constraints=[]
        for i,n in enumerate(self.ik.names):
            low=min(start[n],self.ik.lower[i]+TARGET_JOINT_MARGIN_RAD)
            high=max(start[n],self.ik.upper[i]-TARGET_JOINT_MARGIN_RAD)
            constraints.append({'joint_name':n,'position':(low+high)/2,'tolerance_above':(high-low)/2,
                                'tolerance_below':(high-low)/2,'weight':1.})
        request['path_constraints']={'joint_constraints':constraints}
        goal = MoveGroupGoal()
        genpy.message.fill_message_args(goal.request, [request])
        goal.planning_options.plan_only = True
        diff = goal.planning_options.planning_scene_diff
        diff.is_diff = True
        diff.robot_state.is_diff = True
        diff.world.collision_objects = self.world
        state = deepcopy(self.scene)
        live = self.live()
        js = state.joint_state
        js.position = [start[n] if n in start else live.get(n, v) for n, v in zip(js.name, js.position)]
        state.is_diff = False
        goal.request.start_state = state
        # MoveGroup uses the scene diff for collision geometry during planning;
        # keep the actual measured tool in both the diff and explicit start.
        diff.robot_state.attached_collision_objects = list(state.attached_collision_objects)
        self.move_group.send_goal(goal)
        if not self.move_group.wait_for_result(self.ros.Duration(12)):
            self.move_group.cancel_goal()
            raise CandidateRejected('MoveGroup plan-only deadline exceeded')
        reply = self.move_group.get_result()
        if reply is None or reply.error_code.val != 1:
            raise CandidateRejected('MoveGroup failed, code '+str(reply.error_code.val if reply else None))
        trajectory = reply.planned_trajectory.joint_trajectory
        if set(trajectory.joint_names) != set(start) or len(trajectory.joint_names) != len(start) or not trajectory.points:
            raise CandidateRejected('MoveGroup returned incomplete right-arm trajectory')
        first = dict(zip(trajectory.joint_names, trajectory.points[0].positions))
        if set(first) != set(start) or max(abs(start[n]-first[n]) for n in start) > .01:
            raise CandidateRejected('MoveGroup trajectory starts at a different state')
        knots=[]
        for point in trajectory.points:
            if len(point.positions) != len(start) or not all(map(math.isfinite, point.positions)):
                raise CandidateRejected('Malformed joint trajectory')
            knots.append(dict(zip(trajectory.joint_names, point.positions)))
        path=self.checked_knots(start,target,knots,reject_detour=True)
        self.last_route='moveit_near_joint_goal' if joint_goal is not None else 'moveit_pose_goal'
        return path

    def checked_knots(self,start,target,knots,reject_detour=False):
        previous=start
        path=[]
        self.gaps.clear()
        self.gap_bounds.clear()
        self.check(start)
        line_start=fk_matrix(self.ik.forward(start))[:3,3]
        line=target[:3,3]-line_start
        line_norm=float(line@line)
        lower={n:self.ik.lower[i] for i,n in enumerate(self.ik.names)}
        upper={n:self.ik.upper[i] for i,n in enumerate(self.ik.names)}
        for i,knot in enumerate(knots):
            # The checked and executed joint interpolation between successive
            # knots is monotonic for each joint. Check extrema at the two knots.
            guarded_joint_step(previous,knot,lower,upper,final=i==len(knots)-1)
            for q in joint_path(previous, knot):
                if reject_detour:
                    delta=fk_matrix(self.ik.forward(q))[:3,3]-line_start
                    u=float(np.clip(delta@line/line_norm,0,1)) if line_norm>1e-12 else 0.
                    off_line=float(np.linalg.norm(delta-u*line))
                    if off_line>self.survey.get('max_ompl_detour_m',.20):
                        raise CandidateRejected(f'MoveIt transfer departs {off_line:.3f} m from the direct calibration corridor')
                self.check(q)
                path.append(q)
            previous = knot
        achieved = fk_matrix(self.ik.forward(previous))
        if (np.linalg.norm(achieved[:3,3]-target[:3,3]) > .005
                or angle(achieved[:3,:3].T @ target[:3,:3]) > math.radians(2)):
            raise CandidateRejected('Planned endpoint misses capture gate')
        return path


def existing_samples(directory, plan):
    rows = {p['id']: p for p in plan['poses']}
    seen = set()
    for split in ('train', 'holdout'):
        records = json.loads((directory/(split+'.json')).read_text(encoding='utf-8'))
        for record in records:
            key = record['drive_pose_id']
            if key not in rows or key in seen or rows[key]['split'] != split:
                raise ValueError('Capture contains a repeated, unknown or incorrectly split pose')
            target = transform(rows[key]['base_T_effector'])
            original = transform(record['drive_target_base_T_effector'])
            if not np.allclose(original, target, atol=1e-8, rtol=0):
                raise ValueError('Recorded request differs from the registered volume pose')
            if not (directory/(record['frame_id']+'.npz')).is_file() or not (directory/(record['frame_id']+'.jpg')).is_file():
                raise ValueError('Previously accepted sample is missing its evidence')
            seen.add(key)
    return seen


def capture_gate(driver, gate, plan, requested, timeout):
    item = {k:requested[k] for k in ('id','split','base_T_effector')}
    if gate.exists():
        raise RuntimeError('Outstanding capture request requires reconciliation before movement')
    write_json(gate, item)
    try:
        deadline = time.monotonic()+timeout
        while time.monotonic()<deadline:
            driver.live()
            driver.check_ownership()
            if requested['id'] in existing_samples(gate.parent, plan):
                return
            time.sleep(.12)
        raise RuntimeError('Capture did not acknowledge this stationary pose')
    finally:
        if gate.exists() and json.loads(gate.read_text(encoding='utf-8')) == item:
            gate.unlink()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--scene', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True, help='New append-only JSONL')
    p.add_argument('--limit', type=int, help='Check or collect at most this many remaining registered poses')
    p.add_argument('--first-id', help='Begin read-only diagnosis at this registered ID')
    p.add_argument('--attachment-boxes', type=Path, help='Surveyed rigid board/mount/fingers in right_gripper; absent uses the bounding sphere')
    p.add_argument('--robot-clock-offset-s', type=float, default=0.)
    p.add_argument('--capture-dir', type=Path, help='Active handeye_capture --volume-plan directory')
    p.add_argument('--capture-timeout-s', type=float, default=30.)
    p.add_argument('--drive', action='store_true')
    args = p.parse_args(argv)
    if ((args.limit is not None and args.limit<1) or not -.5<=args.robot_clock_offset_s<=.5
            or not .5<=args.capture_timeout_s<=300 or (args.drive and (not args.capture_dir or args.first_id))):
        p.error('Invalid incremental run, capture timing, or clock offset')
    plan_data = args.plan.read_bytes()
    scene_data = args.scene.read_bytes()
    plan = json.loads(plan_data)
    survey = json.loads(scene_data)
    validate_plan(plan)
    validate_scene(survey)
    args.tool_radius_m = survey['tool_radius_m']
    args.clearance_m = survey['anchor']['tool_clearance_m']
    args.body_clearance_m = survey['anchor']['body_clearance_m']
    pending = list(plan['poses'])
    if args.capture_dir:
        seen = existing_samples(args.capture_dir, plan)
        gate = args.capture_dir/'request.json'
        if args.drive and gate.exists():
            raise RuntimeError('Reconcile outstanding capture gate and measured pose before resuming')
        pending = [row for row in pending if row['id'] not in seen]
    if args.first_id:
        ids = [row['id'] for row in pending]
        if args.first_id not in ids:
            p.error('Unknown or already accepted --first-id')
        pending = pending[ids.index(args.first_id):]
    with args.output.open('x', encoding='utf-8', buffering=1) as audit:
        def record(event, **kw):
            row = dict(event=event, monotonic_s=time.monotonic(), **kw)
            audit.write(json.dumps(row, allow_nan=False)+'\n')
            print(json.dumps(row, allow_nan=False), flush=True)
        record('request', planned_ids=[row['id'] for row in pending], formal_total=len(plan['poses']),
               plan_sha256=hashlib.sha256(plan_data).hexdigest(), scene_sha256=hashlib.sha256(scene_data).hexdigest(),
               drive=args.drive, world_boxes=[b['id'] for b in survey['world_boxes']],
               verified_free_boxes=len(survey['verified_free_sweep_boxes']),
               verified_scene_coverage_boxes=survey.get('verified_scene_coverage_boxes', []))
        if not pending:
            return 0
        driver = SpatialDriver(args, survey)
        driver.record = record
        if not args.first_id:
            pending=nearest_order(driver.anchor,pending,driver.radius)
        pending = pending[:args.limit]
        record('route_order',ids=[row['id'] for row in pending])
        start = driver.anchor_joints
        # A bad geometric start invalidates every outgoing path. Report the
        # collision pair once instead of launching dozens of identical searches.
        try:
            driver.check(start)
        except CandidateRejected as error:
            record('invalid_anchor', reason=str(error), pose=packed(driver.anchor))
            return 2
        failures = []
        for row in pending:
            pose = transform(row['base_T_effector'])
            try:
                visible = marker_projection(pose, survey)
                path = driver.planned_path(start, pose)
                gaps = sorted(driver.gaps)
                record('preflight', id=row['id'], split=row['split'], samples=len(path),route=driver.last_route,
                       target_joints=path[-1],
                       unmeasured_sweep_links=gaps, unmeasured_sweep_bounds_base_m=driver.coverage_report(),
                       marker_visibility=visible)
                if gaps:
                    failures.append(row['id'])
                start = path[-1]
            except CandidateRejected as error:
                failures.append(row['id'])
                record('rejected', id=row['id'], reason=str(error))
        if failures:
            record('incomplete_preflight', failed_ids=failures)
            return 2
        if not args.drive:
            return 0
        import fcntl
        import tempfile
        with open(Path(tempfile.gettempdir())/'baxter-right-handeye-drive.lock','a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
            driver.driving = True
            try:
                for row in pending:
                    if gate.exists():
                        raise RuntimeError('Capture gate became active before movement')
                    measured = driver.live()
                    start = {n:measured[n] for n in driver.joint_names}
                    pose = transform(row['base_T_effector'])
                    path = driver.planned_path(start, pose)
                    if driver.pub is None:
                        driver.enable_publisher()
                    record('dispatch', id=row['id'], sample_count=len(path), start=start)
                    driver.execute(path, start, pose)
                    capture_gate(driver, gate, plan, row, args.capture_timeout_s)
                    record('captured', id=row['id'], split=row['split'])
            finally:
                if driver.pub is not None:
                    try:
                        driver.stop_at_feedback()
                    finally:
                        driver.pub.unregister()
    return 0


if __name__=='__main__':
    raise SystemExit(main())
