"""ROS Noetic bridge. Run on the existing Baxter control PC, bind only to loopback."""
import argparse
import json
import math
import time
import uuid
from contextlib import ExitStack
from pathlib import Path

from .camera import D455Camera
from .contract import ACTION_SPEC, action7, vector
from .rpc import serve


def validate_config(cfg, armed):
    offset=cfg.get('robot_clock_offset_s',0.0)
    if type(offset) not in (int,float) or not math.isfinite(offset) or abs(offset)>2:
        raise ValueError('Measured robot clock offset must be finite and within two seconds')
    if cfg.get('ik_solver', 'baxter_service') not in ('baxter_service', 'kdl_continuous'):
        raise ValueError('Unknown IK solver')
    if cfg["arm"] not in ("left", "right") or not isinstance(cfg.get("camera_serial", ""), str):
        raise ValueError("Set a valid Baxter arm and D455 serial")
    scope = cfg.get("commissioning_scope")
    clearance_test = scope in ('local_clearance_single_step','supervised_cartesian_diagnostic')
    if scope == 'local_clearance_single_step' and cfg.get("motion_budget") != 1:
        raise ValueError("A local clearance test requires exactly one motion budget")
    if scope == 'supervised_cartesian_diagnostic':
        if type(cfg.get('motion_budget')) is not int or not 1 <= cfg['motion_budget'] <= 5:
            raise ValueError('A supervised diagnostic requires a bounded budget of 1..5')
        vector(cfg.get('diagnostic_origin_m'),3,'diagnostic origin')
        radius=cfg.get('diagnostic_radius_m')
        if type(radius) not in (int,float) or not math.isfinite(radius) or not 0 < radius <= .20:
            raise ValueError('Diagnostic travel radius must be at most 20 cm')
        if cfg.get('ik_solver') != 'kdl_continuous':
            raise ValueError('Supervised diagnostic requires continuous IK')
    hybrid_trial = scope == 'supervised_hybrid_task'
    primitive_trial = scope in ('supervised_primitive_task','supervised_hybrid_task')
    task_trial = scope in ('supervised_policy_task','supervised_primitive_task','supervised_hybrid_task')
    if task_trial:
        from .task_guard import validate_guard
        if not primitive_trial or hybrid_trial: validate_guard(cfg['contact_guard'])
        if primitive_trial:
            from .primitives import validate_profile
            from .moveit_guard import options as collision_options
            collision_options(cfg.get('planning', {}))
            validate_profile(cfg['primitives'])
            if armed and cfg['primitives'].get('calibrated') is not True:
                raise ValueError('Primitive geometry must be measured before arming')
        if cfg.get('ik_solver') != 'kdl_continuous' or type(cfg.get('motion_budget')) is not int or not 1 <= cfg['motion_budget'] <= 64:
            raise ValueError('Task trial requires continuous IK and 1..64 actions')
        lease=cfg.get('lease_seconds')
        if type(lease) not in (int,float) or not 1 <= lease <= 600:
            raise ValueError('Task trial requires a lease of at most 600 seconds')
    for key,lo,hi in [('joint_speed_rad_s',.01,.12),('arrival_position_tolerance_m',.002,.01)]:
        value=cfg.get(key,lo)
        if type(value) not in (int,float) or not math.isfinite(value) or not lo <= value <= hi:
            raise ValueError('Invalid '+key)
    from .trajectory import motion_limits
    motion_limits(cfg)
    translation_cap = 0.12 if task_trial else (0.05 if clearance_test else 0.01)
    for key, maximum in (("max_translation_m", translation_cap), ("max_rotation_rad", 0.15 if task_trial else 0.05)):
        value = cfg.get(key, maximum)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= maximum:
            raise ValueError(key + " exceeds the bridge's hard limit")
    if "motion_budget" in cfg and (type(cfg["motion_budget"]) is not int or not 0 <= cfg["motion_budget"] <= (64 if task_trial else 8)):
        raise ValueError("motion_budget must be an integer in 0..8")
    if not armed:
        return
    if cfg.get("commissioned") is not True:
        raise ValueError("Measure the workspace and policy axes, then mark this profile commissioned")
    quat = vector(cfg["policy_to_base_xyzw"], 4, "axis rotation")
    if abs(sum(q*q for q in quat) - 1) > 1e-5:
        raise ValueError("Axis rotation must be a unit quaternion")
    low = vector(cfg["workspace_min_m"], 3, "workspace min")
    high = vector(cfg["workspace_max_m"], 3, "workspace max")
    if any(a >= b for a, b in zip(low, high)):
        raise ValueError("Workspace limits must be ordered")
    age = cfg["max_observation_age_s"]
    if type(age) not in (int, float) or not 0 < age <= 5:
        raise ValueError("max_observation_age_s must lie in (0,5]")


def limit_norm(v, maximum):
    length = math.sqrt(sum(x*x for x in v))
    return [x * maximum / length for x in v] if length > maximum else v


# Engineering acceptance for the supervised local connectivity test. These
# are not the tolerances of a calibrated grasping controller.
CLEARANCE_POSITION_TOLERANCE_M = .01
CLEARANCE_ROTATION_TOLERANCE_RAD = .02
CLEARANCE_JOINT_TOLERANCE_RAD = .0175
CLEARANCE_TRAVEL_MARGIN_M = .015


class Baxter:
    def __init__(self, cfg, armed, *, camera=None):
        import rospy
        import baxter_interface
        from baxter_core_msgs.msg import EndEffectorState, EndpointState, CollisionAvoidanceState, CollisionDetectionState
        from baxter_core_msgs.srv import SolvePositionIK
        from sensor_msgs.msg import Image, JointState

        self.ros = rospy
        self.cfg = cfg
        self.armed = armed
        self.motion_remaining = cfg.get("motion_budget")
        self.lease_end = time.monotonic()+cfg.get('lease_seconds',600)
        self.calibrated_profile = cfg.get("commissioned") is True
        if self.calibrated_profile:
            validate_config(cfg, True)
        self.pending = None
        self.ik = None
        self.limb = baxter_interface.Limb(cfg["arm"])
        if cfg.get('ik_solver') == 'kdl_continuous':
            from .kinematics import ContinuousIK
            self.continuous_ik = ContinuousIK(cfg['arm'])
        # Gripper() sets parameters on construction. A preview must only read telemetry.
        self.effector_state = None
        if armed:
            self.grip = baxter_interface.Gripper(cfg["arm"])
            self.enable = baxter_interface.RobotEnable()
        # The SDK caches the last pose and joints even after their publishers stop.
        self.state_times = {}
        self.state_stamps = {}
        planning = cfg.get('planning')
        self.capture_body_joints = isinstance(planning, dict) and planning.get('collision_backend') == 'moveit'
        self.body_joint_sample = None
        self.wrist_state = None
        self.controller_feedback = {}
        self.subscribers = [
            rospy.Subscriber("/robot/joint_states", JointState, self.note_state, callback_args="joints", queue_size=1),
            rospy.Subscriber("/robot/limb/{}/endpoint_state".format(cfg["arm"]), EndpointState,
                             self.note_state, callback_args="endpoint", queue_size=1),
            rospy.Subscriber("/robot/end_effector/{}_gripper/state".format(cfg["arm"]), EndEffectorState,
                             self.note_effector, queue_size=1),
            rospy.Subscriber("/robot/limb/{}/collision_avoidance_state".format(cfg["arm"]), CollisionAvoidanceState,
                             self.note_controller_feedback, callback_args="avoidance", queue_size=1),
            rospy.Subscriber("/robot/limb/{}/collision_detection_state".format(cfg["arm"]), CollisionDetectionState,
                             self.note_controller_feedback, callback_args="collision", queue_size=1),
        ]
        wrist_topic = cfg.get("wrist_camera_topic", "/cameras/{}_hand_camera/image".format(cfg["arm"]))
        if wrist_topic:
            self.subscribers.append(rospy.Subscriber(wrist_topic, Image, self.note_wrist, queue_size=1, buff_size=4*1024*1024))
        self.owns_camera = camera is None
        self.camera = camera
        self._closed = False
        try:
            if armed:
                self.ready()
            if self.calibrated_profile:
                service = "/ExternalTools/{}/PositionKinematicsNode/IKService".format(cfg["arm"])
                rospy.wait_for_service(service, timeout=5)
                self.ik = rospy.ServiceProxy(service, SolvePositionIK)
            if camera is None:self.camera=D455Camera(cfg.get("camera_serial", ""))
        except BaseException:
            self.close()
            raise

    def close(self):
        """A dual owner closes its shared camera once, after both subscribers stop."""
        if self._closed:return
        self._closed=True
        with ExitStack() as stack:
            if self.owns_camera and self.camera is not None:stack.callback(self.camera.close)
            for subscriber in self.subscribers:stack.callback(subscriber.unregister)

    def health(self):
        from .memory import geometry_fingerprint
        scope = self.cfg.get('commissioning_scope', 'task_workspace')
        capabilities = (['primitives', 'vla'] if scope == 'supervised_hybrid_task' else
                        ['primitives'] if scope == 'supervised_primitive_task' else
                        ['vla'] if scope in ('supervised_policy_task', 'task_workspace') else [])
        return {"armed": self.armed, "can_preview_ik": self.calibrated_profile,
                "backend_capabilities": capabilities,
                "primitive_collision_backend": self.cfg.get('planning', {}).get('collision_backend', 'builtin') if 'primitives' in capabilities else None,
                "hybrid_protocol": 'serialized_backends_v1' if scope == 'supervised_hybrid_task' else None,
                "environment_fingerprint": geometry_fingerprint(self.cfg),
                "ik_solver": self.cfg.get('ik_solver', 'baxter_service'),
                "can_inspect_ik": True,
                "max_action_window": 4,
                "max_action_segment": 16,
                "motion_remaining": getattr(self, "motion_remaining", None),
                "motion_scope": self.cfg.get("commissioning_scope", "task_workspace"),
                "arm": self.cfg["arm"], "action_spec": ACTION_SPEC,
                "policy_to_base_xyzw": self.cfg.get("policy_to_base_xyzw")}

    def ready(self):
        guard = getattr(self, 'primitive_moveit_guard', None)
        if guard is not None and guard.execution_reference is not None:
            guard.check_inactive(guard.execution_reference)
        if time.monotonic() >= getattr(self,'lease_end',math.inf):
            raise RuntimeError('Motion lease expired')
        state = self.enable.state()
        if self.ros.is_shutdown() or not state or not state.enabled or state.stopped or state.error:
            raise RuntimeError("Baxter must already be enabled and clear of stops/errors")
        if self.grip.type() != "electric" or not self.grip.calibrated() or self.grip.error():
            raise RuntimeError("Expected a calibrated electric gripper (type={}, calibrated={}, error={})".format(
                self.grip.type(), self.grip.calibrated(), self.grip.error()))

    def telemetry_time(self, stamp, key):
        # Independent robot/host clocks can differ by a few milliseconds. Never
        # extend freshness into the future, or refresh it with a repeated stamp.
        # Explicit measured robot-minus-host offset, not a relaxed freshness limit.
        age = (self.ros.Time.now() - stamp).to_sec() + self.cfg.get('robot_clock_offset_s',0.0)
        seconds = stamp.to_sec()
        if (not math.isfinite(age) or not math.isfinite(seconds) or
                not -0.1 <= age < 0.5 or seconds <= self.state_stamps.get(key, -math.inf)):
            return None
        self.state_stamps[key] = seconds
        return time.monotonic() - max(0.0, age)

    def note_state(self, message, key):
        if key == "joints" and not set(self.limb.joint_names()).issubset(message.name):
            return
        received = self.telemetry_time(message.header.stamp, key)
        if received is not None:
            self.state_times[key] = received
            if key == 'joints' and getattr(self, 'capture_body_joints', False):
                names = message.name
                if (len(names) == len(message.position) == len(set(names))
                        and all(math.isfinite(x) for x in message.position)):
                    self.body_joint_sample = (received, dict(zip(names, message.position)))
                else:
                    self.body_joint_sample = None

    def full_body_joint_sample(self):
        if self.body_joint_sample is None:
            raise RuntimeError('No complete measured Baxter joint sample for MoveIt')
        return self.body_joint_sample

    def note_effector(self, message):
        received = self.telemetry_time(message.timestamp, "gripper")
        if received is not None:
            self.effector_state = (received, message)

    def note_wrist(self, message):
        received = self.telemetry_time(message.header.stamp, "wrist")
        if received is not None:
            self.wrist_state = (received, message)

    def note_controller_feedback(self,message,key):
        received=self.telemetry_time(message.header.stamp,'controller_'+key)
        if received is not None:
            value=({'other_arm':bool(message.other_arm),'objects':list(message.collision_object)}
                   if key=='avoidance' else bool(message.collision_state))
            self.controller_feedback[key]=(received,value)

    def check_controller_feedback(self):
        from .trajectory import check_controller_feedback,ControllerSafetyError
        try:check_controller_feedback(getattr(self,'controller_feedback',{}),time.monotonic())
        except ControllerSafetyError as error:
            self.audit('factory_controller_rejected',error=str(error),
                       context=getattr(self,'motion_context',{}))
            raise

    def joints(self):
        now = time.monotonic()
        if any(now - self.state_times.get(key, -math.inf) >= 0.5 for key in ("joints", "endpoint")):
            raise RuntimeError("Stale Baxter joint or endpoint telemetry")
        angles = self.limb.joint_angles()
        names = self.limb.joint_names()
        if len(names) != 7 or set(names) != set(angles):
            raise RuntimeError("Incomplete Baxter joint state")
        vector([angles[n] for n in names], 7, "joint angles")
        return angles

    def read_state(self, frame_number=None):
        """Fresh arm telemetry without acquiring another frame or operating a gripper."""
        angles = self.joints()
        pose = self.limb.endpoint_pose()
        position = pose["position"]
        effector = self.effector_state
        if effector is None or time.monotonic() - effector[0] >= 0.5:
            raise RuntimeError("Stale Baxter gripper telemetry")
        grip = effector[1]
        gripping = None if grip.gripping == grip.STATE_UNKNOWN else grip.gripping == grip.STATE_TRUE
        state = {"arm": self.cfg["arm"], "frame_number": frame_number, "eef_m": [position.x, position.y, position.z],
                 "gripper_percent": grip.position, "gripping": gripping}
        state['gripper_force']=getattr(grip,'force',None)
        if "orientation" in pose:
            q = pose["orientation"]
            state["eef_xyzw"] = [q.x, q.y, q.z, q.w]
        names = self.limb.joint_names()
        state["joint_names"] = names
        state["joint_rad"] = [angles[n] for n in names]
        return state

    def observe(self):
        began=time.monotonic()
        self.pending = None
        image, captured, frame_number = self.camera.capture()
        after_camera=time.monotonic()
        obs_id = uuid.uuid4().hex
        state=self.read_state(frame_number)
        angles=dict(zip(state['joint_names'],state['joint_rad']))
        obs = {"id": obs_id, "image": image, "state": state}
        after_state=time.monotonic()
        wrist = getattr(self, "wrist_state", None)
        if wrist is not None and time.monotonic() - wrist[0] < 0.5:
            import cv2
            from cv_bridge import CvBridge
            from .contract import image_data
            pixels = CvBridge().imgmsg_to_cv2(wrist[1], desired_encoding="bgr8")
            ok, jpeg = cv2.imencode(".jpg", pixels, [cv2.IMWRITE_JPEG_QUALITY, 90])
            if not ok:
                raise RuntimeError("Cannot encode wrist image")
            obs["wrist_image"] = image_data(jpeg.tobytes())
            state["wrist_age_s"] = round(time.monotonic() - wrist[0], 4)
        elif self.cfg.get("require_wrist", False):
            raise RuntimeError("Fresh real wrist camera image required by Pi05")
        self.pending = (obs_id, captured, angles)
        ended=time.monotonic()
        obs['camera_metadata']={'scene':{'source':'d455','frame_number':frame_number,
                                        'age_s':round(max(0.,ended-captured),4)}}
        if 'wrist_image' in obs:
            obs['camera_metadata']['wrist']={'source':self.cfg['arm']+'_wrist','arm':self.cfg['arm'],
                                            'frame_number':getattr(wrist[1].header,'seq',None),
                                            'age_s':round(ended-wrist[0],4)}
        obs['timings_s']={'camera_capture':after_camera-began,'state_read':after_state-after_camera,
                          'wrist_image':ended-after_state,'baxter_observe_total':ended-began}
        return obs

    def scene_snapshot(self):
        obs=self.observe()
        folder=Path('runs/geometry');folder.mkdir(parents=True,exist_ok=True)
        path=folder/(obs['id']+'.npz')
        self.camera.save_depth(str(path))
        return {'obs_id':obs['id'],'state':obs['state'],'depth_file':str(path.resolve())}

    def inside(self, xyz):
        if not all(a <= p <= b for p, a, b in zip(xyz, self.cfg["workspace_min_m"], self.cfg["workspace_max_m"])):
            raise ValueError("Endpoint outside commissioned workspace")
        if self.cfg.get('commissioning_scope') == 'supervised_cartesian_diagnostic':
            origin=self.cfg['diagnostic_origin_m']
            if math.sqrt(sum((a-b)**2 for a,b in zip(xyz,origin))) > self.cfg['diagnostic_radius_m']:
                raise ValueError('Endpoint outside cumulative diagnostic travel radius')

    def target(self, action, *, enforce_workspace=True, project_table=False):
        import numpy as np
        from geometry_msgs.msg import PoseStamped
        from tf import transformations as tf

        pose = self.limb.endpoint_pose()
        if enforce_workspace:
            self.guard_pose(pose)
        pos, quat = pose["position"], pose["orientation"]
        xyz = vector([pos.x, pos.y, pos.z], 3, "current position")
        orientation = vector([quat.x, quat.y, quat.z, quat.w], 4, "current orientation")
        if abs(sum(q*q for q in orientation) - 1.0) > 0.001:
            raise ValueError("Invalid observed endpoint orientation")
        if enforce_workspace:
            self.inside(xyz)
        axis = self.cfg["policy_to_base_xyzw"]
        delta = tf.quaternion_matrix(axis)[:3, :3].dot(limit_norm(action[:3], self.cfg.get("max_translation_m", 0.01)))
        new_xyz = np.asarray(xyz) + delta
        rotation = tf.quaternion_from_euler(*limit_norm(action[3:6], self.cfg.get("max_rotation_rad", 0.05)))
        base_rotation = tf.quaternion_multiply(tf.quaternion_multiply(axis, rotation), tf.quaternion_inverse(axis))
        new_quat = tf.quaternion_multiply(base_rotation, orientation)
        if project_table and self.cfg.get('contact_guard'):
            from .task_guard import contact_bounds
            floor=contact_bounds(self.cfg['contact_guard'],list(map(float,new_quat)))['minimum_eef_z_m']
            new_xyz[2]=max(new_xyz[2],floor+.003)
        if enforce_workspace:
            self.inside(new_xyz)
        target = PoseStamped()
        target.header.frame_id = "base"
        target.header.stamp = self.ros.Time.now()
        p, q = target.pose.position, target.pose.orientation
        p.x, p.y, p.z = map(float, new_xyz)
        q.x, q.y, q.z, q.w = map(float, new_quat)
        if enforce_workspace:
            self.guard_pose({'position':p,'orientation':q})
        return target

    def solve(self, target, current):
        if self.cfg.get('ik_solver') == 'kdl_continuous':
            return self.continuous_ik.solve(target, current, self.limb.endpoint_pose())
        from baxter_core_msgs.srv import SolvePositionIKRequest

        request = SolvePositionIKRequest()
        request.pose_stamp = [target]
        request.seed_mode = request.SEED_CURRENT
        result = self.ik(request)
        if not result.result_type or not result.joints:
            raise RuntimeError("IK returned no solution")
        code = result.result_type[0]
        if (ord(code) if isinstance(code, (str, bytes)) else code) == result.RESULT_INVALID:
            raise RuntimeError("Baxter rejected IK target")
        joint = result.joints[0]
        positions = vector(list(joint.position), 7, "IK solution")
        solution = dict(zip(joint.name, positions))
        if set(solution) != set(current) or max(abs(solution[k]-current[k]) for k in current) > 0.15:
            raise ValueError("Incomplete or discontinuous IK solution")
        return solution

    def inspect_action(self, obs_id, spec, values):
        """Read-only IK diagnostic; an IK solution is not permission to move."""
        pending, self.pending = self.pending, None
        if spec != ACTION_SPEC or pending is None or obs_id != pending[0]:
            raise ValueError("Missing current observation or wrong action specification")
        if self.ik is None and self.cfg.get('ik_solver') != 'kdl_continuous':
            from baxter_core_msgs.srv import SolvePositionIK
            service = "/ExternalTools/{}/PositionKinematicsNode/IKService".format(self.cfg["arm"])
            self.ros.wait_for_service(service, timeout=5)
            self.ik = self.ros.ServiceProxy(service, SolvePositionIK)
        current = self.joints()
        action = action7(values)
        target = self.target(action, enforce_workspace=self.calibrated_profile)
        solution = self.solve(target, current)
        if time.monotonic()-pending[1] > self.cfg["max_observation_age_s"]:
            raise RuntimeError("Observation expired during IK inspection")
        if max(abs(current[k]-v) for k,v in pending[2].items()) > 0.015:
            raise RuntimeError("Arm moved since observation")
        p, q = target.pose.position, target.pose.orientation
        return {"obs_id": obs_id, "applied": False, "motion_allowed": False,
                "workspace_verified": self.calibrated_profile,
                "target_base_m": [p.x,p.y,p.z], "target_xyzw": [q.x,q.y,q.z,q.w],
                "joint_targets_rad": solution, "gripper_open": action[6] >= 0.5,
                "translation_limit_m": self.cfg.get("max_translation_m",0.01),
                "rotation_limit_rad": self.cfg.get("max_rotation_rad",0.05)}

    def move(self, solution, target=None):
        if target is None:
            raise ValueError('Every motion requires a Cartesian arrival target')
        began = time.monotonic()
        initial = self.joints()
        speed = self.cfg.get('joint_speed_rad_s', .06)
        duration = max(.35, 1.875*max(abs(solution[n]-initial[n]) for n in initial)/speed)
        deadline = began + duration + 2.0
        start = self.endpoint_position()
        while time.monotonic() < deadline:
            self.ready()
            self.check_controller_feedback()
            now = self.joints()
            position = self.endpoint_position()
            if math.sqrt(sum((a-b)**2 for a,b in zip(position,start))) > self.cfg.get('max_translation_m',.01)+CLEARANCE_TRAVEL_MARGIN_M:
                raise RuntimeError('Endpoint exceeded per-command travel envelope')
            self.inside(position)
            self.guard_pose(self.limb.endpoint_pose())
            errors = {k:solution[k]-now[k] for k in now}
            if time.monotonic()-began >= duration and max(abs(x) for x in errors.values()) < CLEARANCE_JOINT_TOLERANCE_RAD:
                distance, angle = self.arrival_error(target)
                if distance < self.cfg.get('arrival_position_tolerance_m',.01) and angle < CLEARANCE_ROTATION_TOLERANCE_RAD:
                    return
            phase = min(1.0,(time.monotonic()-began)/duration)
            fraction = 10*phase**3-15*phase**4+6*phase**5
            command = {k:initial[k]+fraction*(solution[k]-initial[k]) for k in initial}
            if max(abs(command[k]-now[k]) for k in now) > .05:
                raise RuntimeError('Joint tracking error exceeded limit')
            self.limb.set_joint_positions(command)
            time.sleep(.01)
        raise RuntimeError('Joint target did not settle')

    def guard_pose(self, pose):
        if self.cfg.get('commissioning_scope') in ('supervised_primitive_task','supervised_hybrid_task'):
            # Both backends obey the intersection of the commissioned geometries.
            from .primitives import tcp_from_eef, check_tcp
            p, q = pose['position'], pose['orientation']
            profile = self.cfg['primitives']
            tcp = tcp_from_eef([p.x,p.y,p.z], [q.x,q.y,q.z,q.w], profile)
            check_tcp(tcp,profile)
        guard = self.cfg.get('contact_guard')
        if guard:
            from .task_guard import check_contact_pose
            p,q=pose['position'],pose['orientation']
            return check_contact_pose(guard,[p.x,p.y,p.z],[q.x,q.y,q.z,q.w],enforce_planning_margin=False)

    def arrival_error(self, target):
        pose = self.limb.endpoint_pose()
        a,b = pose['position'],target.pose.position
        position_error = math.sqrt((a.x-b.x)**2+(a.y-b.y)**2+(a.z-b.z)**2)
        a,b = pose['orientation'],target.pose.orientation
        observed = vector([a.x,a.y,a.z,a.w],4,'measured orientation')
        expected = vector([b.x,b.y,b.z,b.w],4,'target orientation')
        if abs(sum(v*v for v in observed)-1) > .001:
            raise RuntimeError('Invalid measured orientation at arrival')
        cosine = abs(sum(x*y for x,y in zip(observed,expected)))
        return position_error, 2*math.acos(min(1.0,cosine))

    def endpoint_position(self):
        p = self.limb.endpoint_pose()["position"]
        return vector([p.x,p.y,p.z], 3, "measured endpoint")

    def audit(self, event, **fields):
        path = self.cfg.get("motion_audit_path")
        if path:
            with Path(path).open("a", encoding="utf-8") as output:
                output.write(json.dumps(dict(event=event, time=time.time(), **fields), allow_nan=False)+"\n")
                output.flush()

    def prepare_target(self, obs_id, spec, values):
        pending, self.pending = self.pending, None
        if not self.calibrated_profile or spec != ACTION_SPEC:
            raise RuntimeError("Missing commissioned profile or action contract differs")
        if pending is None or obs_id != pending[0]:
            raise ValueError("Observation already consumed or superseded")
        current = self.joints()
        action = action7(values)
        target = self.target(action)
        solution = self.solve(target, current)
        if time.monotonic() - pending[1] > self.cfg["max_observation_age_s"]:
            raise RuntimeError("Observation expired before movement")
        before_motion = self.joints()
        if max(abs(before_motion[k]-v) for k, v in pending[2].items()) > 0.015:
            raise RuntimeError("Arm moved since observation")
        return action, target, solution

    def preview(self, obs_id, spec, values):
        action, target, solution = self.prepare_target(obs_id, spec, values)
        p = target.pose.position
        q = target.pose.orientation
        return {"obs_id": obs_id, "applied": False, "frame": target.header.frame_id,
                "xyz_m": [p.x, p.y, p.z], "xyzw": [q.x, q.y, q.z, q.w],
                "joint_targets_rad": solution, "open_gripper": action[6] >= 0.5}

    def step(self, obs_id, spec, values):
        if not self.armed:
            raise RuntimeError("Bridge is read-only")
        remaining = getattr(self, "motion_remaining", None)
        if remaining is not None and remaining <= 0:
            raise RuntimeError("Robot-side motion budget exhausted")
        self.ready()
        action, target, solution = self.prepare_target(obs_id, spec, values)
        if remaining is not None:
            self.motion_remaining -= 1
        joints_before = self.joints()
        endpoint_before = self.endpoint_position()
        p = target.pose.position
        self.audit("dispatch", obs_id=obs_id, endpoint_before_m=endpoint_before,
                   endpoint_target_m=[p.x,p.y,p.z], joint_target_rad=solution)
        try:
            self.move(solution, target)
            self.ready()
            if not self.grip.command_position(100.0 if action[6] >= 0.5 else 0.0, block=True, timeout=2):
                raise RuntimeError("Gripper command timed out")
        except BaseException as error:
            try:
                self.limb.set_joint_positions(self.joints())
            except Exception as exc:
                self.ros.logwarn("Unable to hold current joints: %s", exc)
            try:
                self.grip.stop(block=False)
            except Exception as exc:
                self.ros.logwarn("Unable to stop gripper: %s", exc)
            try:
                self.audit("failed", obs_id=obs_id, error=str(error), endpoint_after_m=self.endpoint_position(),
                           joint_after_rad=self.joints())
            except Exception as exc:
                self.ros.logwarn("Unable to record failed action: %s", exc)
            raise
        endpoint_after = self.endpoint_position()
        receipt = {"obs_id": obs_id, "applied": True, "gripper_percent": self.grip.position(),
                "endpoint_before_m": endpoint_before, "endpoint_after_m": endpoint_after,
                "endpoint_target_m": [p.x,p.y,p.z],
                "measured_translation_m": math.sqrt(sum((a-b)**2 for a,b in zip(endpoint_after,endpoint_before))),
                "joint_before_rad": joints_before, "joint_after_rad": self.joints(),
                "joint_target_rad": solution, "motion_remaining": getattr(self,"motion_remaining",None)}
        receipt['position_error_m'],receipt['rotation_error_rad'] = self.arrival_error(target)
        receipt['target_reached'] = (receipt['position_error_m'] < self.cfg.get('arrival_position_tolerance_m',.01) and
                                     receipt['rotation_error_rad'] < CLEARANCE_ROTATION_TOLERANCE_RAD)
        receipt['arrival_position_tolerance_m'] = self.cfg.get('arrival_position_tolerance_m',.01)
        receipt['arrival_rotation_tolerance_rad'] = CLEARANCE_ROTATION_TOLERANCE_RAD
        receipt['gripping'] = bool(self.grip.gripping())
        self.audit("applied", **receipt)
        return receipt

    def step_chunk(self, obs_id, spec, actions):
        if not isinstance(actions,list) or not 1 <= len(actions) <= 4:
            raise ValueError('Expected a bounded action window of 1..4')
        actions=[action7(a) for a in actions]
        if self.cfg.get('commissioning_scope') not in ('supervised_policy_task','supervised_hybrid_task'):
            raise RuntimeError('Action windows require a bounded task profile')
        if self.pending is None or self.pending[0] != obs_id:
            raise ValueError('Window requires a current unconsumed observation')
        if self.motion_remaining is None or self.motion_remaining < len(actions):
            raise RuntimeError('Insufficient robot-side action budget')
        if sum(math.sqrt(sum(v*v for v in a[:3])) for a in actions) > .08:
            raise ValueError('Predicted window exceeds 8 cm path-length cap')
        captured=self.pending[1]
        receipts=[]
        reobserve_reason='window_complete'
        started=time.monotonic()
        try:
            for index,action in enumerate(actions):
                if index and time.monotonic()-captured > self.cfg['max_observation_age_s']-.25:
                    # End a time-stretched window early and ask for a NEW image.
                    reobserve_reason='observation_deadline'
                    break
                token=obs_id if index==0 else obs_id+':window:'+str(index)
                if index:
                    # This is still the SAME camera observation. Preserve its
                    # capture time; do not pretend that a later waypoint has a new image.
                    self.pending=(token,captured,self.joints())
                receipt=self.step(token,spec,action)
                receipts.append(receipt)
                if receipt.get('target_reached') is False:
                    raise RuntimeError('Window stopped after a waypoint missed its target')
                # A gripper transition needs a new image before another action.
                if index+1<len(actions) and (actions[index+1][6]>=.5)!=(action[6]>=.5):
                    reobserve_reason='gripper_transition'
                    break
        except BaseException as exc:
            self.pending=None
            self.audit('window_failed',obs_id=obs_id,error=str(exc),applied_actions=len(receipts))
            raise
        self.pending=None
        return {'obs_id':obs_id,'applied':True,'target_reached':True,'receipts':receipts,
                'applied_actions':len(receipts),'seconds':time.monotonic()-started,
                'reobserve_reason':reobserve_reason,
                'execution_mode':'time_scaled_waypoints','motion_remaining':self.motion_remaining}

    def prepare_segment(self, obs_id, spec, actions):
        from copy import deepcopy
        from tf import transformations as tf
        from .segments import compose_prefix
        from .task_guard import contact_bounds
        pending,self.pending=self.pending,None
        if self.cfg.get('commissioning_scope') not in ('supervised_policy_task','supervised_hybrid_task') or not self.calibrated_profile:
            raise RuntimeError('Policy segments require a commissioned task profile')
        if spec != ACTION_SPEC or pending is None or pending[0] != obs_id:
            raise ValueError('Segment requires the current unconsumed observation')
        effector=self.effector_state
        if effector is None or time.monotonic()-effector[0]>=.5:
            raise RuntimeError('Stale gripper telemetry before segment planning')
        action,used,reason,path_length=compose_prefix(actions,effector[1].position>70)
        if self.motion_remaining is None or self.motion_remaining < used:
            raise RuntimeError('Insufficient robot-side policy budget')
        if self.armed:self.ready()
        start=self.limb.endpoint_pose()
        target=self.target(action,project_table=True)
        p,q=start['position'],start['orientation']
        a=[p.x,p.y,p.z];qa=[q.x,q.y,q.z,q.w]
        p,q=target.pose.position,target.pose.orientation
        b=[p.x,p.y,p.z];qb=[q.x,q.y,q.z,q.w]
        distance=math.dist(a,b)
        if distance > self.cfg['max_translation_m']+1e-6:
            raise ValueError('Table projection exceeds the segment travel budget')
        angle=2*math.acos(min(1.,abs(sum(x*y for x,y in zip(qa,qb)))))
        count=max(1,math.ceil(distance/.01),math.ceil(angle/.015))
        initial=self.joints();seed=initial;observed=start;path=[]
        for index in range(1,count+1):
            fraction=index/count
            point=deepcopy(target)
            xyz=[x+(y-x)*fraction for x,y in zip(a,b)]
            quat=tf.quaternion_slerp(qa,qb,fraction)
            if self.cfg.get('contact_guard'):
                floor=contact_bounds(self.cfg['contact_guard'],list(map(float,quat)))['minimum_eef_z_m']
                xyz[2]=max(xyz[2],floor+.003)
            self.inside(xyz)
            if math.dist(a,xyz)>self.cfg['max_translation_m']+1e-6:
                raise ValueError('Intermediate table projection exceeds segment travel budget')
            point.pose.position.x,point.pose.position.y,point.pose.position.z=xyz
            point.pose.orientation.x,point.pose.orientation.y,point.pose.orientation.z,point.pose.orientation.w=map(float,quat)
            self.guard_pose({'position':point.pose.position,'orientation':point.pose.orientation})
            try:
                solution=self.continuous_ik.solve(point,seed,observed)
            except ValueError as exc:
                if used<=1 or not str(exc).startswith(('No continuous','KDL output exceeds')):
                    raise
                # Keep a feasible prefix of this prediction, never jump to a
                # distant IK branch to force the full requested displacement.
                self.pending=pending
                result=self.prepare_segment(obs_id,spec,actions[:used-1])
                return result[0],result[1],'ik_horizon',result[3],result[4],result[5]
            path.append((solution,point))
            seed=solution;observed={'position':point.pose.position,'orientation':point.pose.orientation}
        if time.monotonic()-pending[1] > self.cfg['max_observation_age_s']:
            raise RuntimeError('Observation expired during segment planning')
        if max(abs(self.joints()[k]-v) for k,v in pending[2].items())>.015:
            raise RuntimeError('Arm moved since segment observation')
        return action,used,reason,path_length,initial,path

    def preview_segment(self,obs_id,spec,actions):
        action,used,reason,length,initial,path=self.prepare_segment(obs_id,spec,actions)
        p=path[-1][1].pose.position
        return {'obs_id':obs_id,'applied':False,'source_actions':used,'reason':reason,
                'composed_action':action,'source_path_m':length,'waypoints':len(path),
                'endpoint_target_m':[p.x,p.y,p.z]}

    def move_segment(self,initial,path,start,max_travel=None,*,continuous=False,motion_phase=None):
        """Execute checked paths; continuous interpolation requires joint-box/FK preflight."""
        from .trajectory import JointTrajectory,motion_limits
        began=time.monotonic();speed,acceleration=motion_limits(self.cfg,motion_phase)
        schedule=[];elapsed=0.;previous=initial
        for solution,target in path:
            duration=max(.15,1.875*max(abs(solution[k]-previous[k]) for k in initial)/speed)
            schedule.append((elapsed,duration,previous,solution))
            elapsed+=duration;previous=solution
        if not schedule:raise ValueError('Empty motion path')
        legacy_seconds=elapsed
        # Half the acceleration budget remains for feedback-driven clock changes.
        trajectory=JointTrajectory(initial,[q for q,_ in path],speed,acceleration*.5) if continuous else None
        if trajectory is not None:elapsed=trajectory.duration
        knots=[initial]+[q for q,_ in path]
        joint_travel={n:sum(abs(b[n]-a[n]) for a,b in zip(knots,knots[1:])) for n in initial}
        try:
            xyz=[list(start)]+[[float(getattr(target.pose.position,n)) for n in ('x','y','z')] for _,target in path]
            planned_eef_path=sum(math.dist(a,b) for a,b in zip(xyz,xyz[1:]))
        except (TypeError,ValueError):planned_eef_path=None
        measured_path=0.;last_position=list(start);samples=0;peak_command_speed=0.
        timings={name:0. for name in ('controller_checks','feedback_read','geometry_checks','interpolation',
                                    'command_publish','arrival_check','sleep')}
        build_seconds=time.monotonic()-began
        context=dict(getattr(self,'motion_context',{}));trajectory_id=uuid.uuid4().hex
        def audit(event,**fields):
            self.audit(event,trajectory_id=trajectory_id,context=context,motion_phase=motion_phase,**fields)
        audit('segment_started',scheduled_seconds=elapsed,legacy_schedule_seconds=legacy_seconds,
              continuous=continuous,waypoints=len(path),joint_speed_rad_s=speed,
              joint_acceleration_rad_s2=acceleration,build_seconds=build_seconds,
              trajectory_metadata=trajectory.metadata if trajectory is not None else None)
        loop_began=time.monotonic();last_tick=loop_began;progress=0.;index=0;settled=None
        command=dict(initial);paused_since=None;retimed_seconds=0.;peak_tracking=0.
        scale=1.;feedback_delay=0.;loop_delay=0.;end_wall=None;last_audit=loop_began
        deadline=elapsed+min(60.,max(10.,elapsed))+2.5
        def metrics():
            loop_elapsed=time.monotonic()-loop_began
            return {'trajectory_id':trajectory_id,'context':context,'motion_phase':motion_phase,
                    'continuous':continuous,'scheduled_seconds':elapsed,'legacy_schedule_seconds':legacy_seconds,
                    'trajectory_seconds':progress,'elapsed_s':loop_elapsed,
                    'retimed_seconds':retimed_seconds,'feedback_slowdown_s':feedback_delay,
                    'loop_delay_s':loop_delay,'settling_s':0. if end_wall is None else time.monotonic()-end_wall,
                    'peak_tracking_rad':peak_tracking,'peak_command_speed_rad_s':peak_command_speed,
                    'raw_measured_eef_path_m':measured_path,'path_note':'Raw telemetry path includes measurement jitter',
                    'command_samples':samples,
                    'build_seconds':build_seconds,'loop_timings_s':dict(timings),
                    'unattributed_loop_s':max(0.,loop_elapsed-sum(timings.values())),
                    'timing_note':'Loop sections are disjoint; slowdown, delay and settling are explanatory overlaps, not extra costs',
                    'planned_eef_path_m':planned_eef_path,'joint_travel_rad':joint_travel,
                    'bottleneck_joint':max(joint_travel,key=joint_travel.get),
                    'joint_speed_limit_rad_s':speed,'joint_acceleration_limit_rad_s2':acceleration,
                    'waypoints':len(path),'trajectory_metadata':trajectory.metadata if trajectory is not None else None}
        try:
            while time.monotonic()-loop_began < deadline:
                stamp=time.monotonic();self.ready();self.check_controller_feedback()
                timings['controller_checks']+=time.monotonic()-stamp
                stamp=time.monotonic();now=self.joints();position=self.endpoint_position()
                measured_path+=math.dist(position,last_position);last_position=position
                timings['feedback_read']+=time.monotonic()-stamp
                stamp=time.monotonic();self.inside(position);self.guard_pose(self.limb.endpoint_pose())
                if math.dist(position,start)>(self.cfg['max_translation_m'] if max_travel is None else max_travel)+CLEARANCE_TRAVEL_MARGIN_M:
                    raise RuntimeError('Endpoint exceeded segment travel envelope')
                timings['geometry_checks']+=time.monotonic()-stamp
                wall=time.monotonic();t=wall-loop_began
                tracking={k:command[k]-now[k] for k in initial}
                worst=max(tracking,key=lambda k:abs(tracking[k]))
                lead=abs(tracking[worst]);peak_tracking=max(peak_tracking,lead)
                if lead>.05:
                    audit('segment_tracking_rejected',joint=worst,joint_errors_rad=tracking,
                          command_rad=command,measured_rad=now,elapsed_s=t,
                          trajectory_seconds=progress,scheduled_seconds=elapsed,endpoint_m=position)
                    raise RuntimeError('Joint tracking error exceeded segment limit ({}: {:.4f} rad)'.format(worst,tracking[worst]))
                dt=max(0.,wall-last_tick);last_tick=wall;effective=min(.05,dt)
                desired=min(1.,max(0.,(.035-lead)/.015))
                if trajectory is None:scale=desired
                else:
                    slew=acceleration/(2*max(trajectory.peak_velocity,1e-9))*effective
                    scale+=min(slew,max(-slew,desired-scale))
                advance=effective*scale
                if progress<elapsed:
                    feedback_delay+=effective-advance;loop_delay+=dt-effective
                    retimed_seconds+=dt-advance
                progress=min(elapsed,progress+advance)
                if scale<.01 and progress<elapsed:
                    if paused_since is None:paused_since=wall
                    if wall-paused_since>=8.:
                        audit('segment_following_stalled',joint=worst,joint_errors_rad=tracking,
                              command_rad=command,measured_rad=now,elapsed_s=t,trajectory_seconds=progress,endpoint_m=position)
                        raise RuntimeError('Joint following stalled while trajectory waited for '+worst)
                else:paused_since=None
                stamp=time.monotonic();previous_command=command
                if trajectory is not None:command=trajectory.sample(progress)
                else:
                    while index+1<len(schedule) and progress>=schedule[index+1][0]:index+=1
                    offset,duration,previous,solution=schedule[index]
                    phase=min(1.,max(0.,(progress-offset)/duration))
                    fraction=10*phase**3-15*phase**4+6*phase**5
                    command={k:previous[k]+fraction*(solution[k]-previous[k]) for k in initial}
                timings['interpolation']+=time.monotonic()-stamp
                if dt>0:peak_command_speed=max(peak_command_speed,max(abs(command[k]-previous_command[k])/dt for k in initial))
                tracking={k:command[k]-now[k] for k in initial}
                worst=max(tracking,key=lambda k:abs(tracking[k]))
                if abs(tracking[worst])>.05:
                    audit('segment_tracking_rejected',joint=worst,joint_errors_rad=tracking,
                          command_rad=command,measured_rad=now,elapsed_s=t,trajectory_seconds=progress,endpoint_m=position)
                    raise RuntimeError('Joint tracking error exceeded segment limit ({}: {:.4f} rad)'.format(worst,tracking[worst]))
                stamp=time.monotonic();self.limb.set_joint_positions(command);samples+=1
                timings['command_publish']+=time.monotonic()-stamp
                if wall-last_audit>=1.:
                    audit('segment_progress',elapsed_s=t,trajectory_seconds=progress,tracking_rad=lead,
                          clock_scale=scale,endpoint_m=position,feedback_slowdown_s=feedback_delay,loop_delay_s=loop_delay)
                    last_audit=wall
                if progress>=elapsed:
                    if end_wall is None:end_wall=wall
                    stamp=time.monotonic();distance,angle=self.arrival_error(path[-1][1])
                    reached=(distance<self.cfg.get('arrival_position_tolerance_m',.01) and angle<.04)
                    settled=(settled if settled is not None else time.monotonic()) if reached else None
                    timings['arrival_check']+=time.monotonic()-stamp
                    if settled is not None and time.monotonic()-settled>=.2:
                        result=metrics();self.audit('segment_arrived',**result);return result
                stamp=time.monotonic();time.sleep(.01);timings['sleep']+=time.monotonic()-stamp
            raise RuntimeError('Segment did not arrive within bounded following/settling time')
        except BaseException as error:
            result=metrics();result['error']=str(error)
            self.audit('segment_execution_failed',**result)
            raise

    def step_segment(self,obs_id,spec,actions):
        if not self.armed:raise RuntimeError('Bridge is read-only')
        action,used,reason,length,initial,path=self.prepare_segment(obs_id,spec,actions)
        self.motion_remaining-=used
        start=self.endpoint_position();began=time.monotonic();p=path[-1][1].pose.position
        self.audit('segment_dispatch',obs_id=obs_id,source_actions=actions[:used],composed_action=action,
                   endpoint_before_m=start,endpoint_target_m=[p.x,p.y,p.z],waypoints=len(path))
        try:
            self.move_segment(initial,path,start)
            self.ready()
            if not self.grip.command_position(100. if action[6]>=.5 else 0.,block=True,timeout=2):
                raise RuntimeError('Gripper command timed out')
            time.sleep(.25)
            self.ready()
            self.guard_pose(self.limb.endpoint_pose())
        except BaseException as exc:
            try:self.limb.set_joint_positions(self.joints());self.grip.stop(block=False)
            except Exception as stop_error:self.ros.logwarn('Segment stop failed: %s',stop_error)
            self.audit('segment_failed',obs_id=obs_id,error=str(exc),endpoint_after_m=self.endpoint_position())
            raise
        distance,angle=self.arrival_error(path[-1][1]);end=self.endpoint_position()
        receipt={'obs_id':obs_id,'applied':True,'applied_actions':used,'execution_mode':'policy_segment',
                 'target_reached':distance<self.cfg.get('arrival_position_tolerance_m',.01) and angle<.04,
                 'endpoint_before_m':start,'endpoint_after_m':end,'endpoint_target_m':[p.x,p.y,p.z],
                 'composed_action':action,'source_path_m':length,'waypoints':len(path),
                 'measured_translation_m':math.dist(start,end),'position_error_m':distance,
                 'rotation_error_rad':angle,'gripper_percent':self.grip.position(),'gripping':bool(self.grip.gripping()),
                 'reobserve_reason':reason,'seconds':time.monotonic()-began,'motion_remaining':self.motion_remaining}
        self.audit('segment_applied',**receipt)
        return receipt


def build_rpc_methods(robot):
    """One serialized bridge owns both backends; never run competing robot writers."""
    scope = robot.cfg.get('commissioning_scope')
    primitives = None
    methods = {'healthz': robot.health, 'env.observe': robot.observe,
               'env.scene_snapshot': robot.scene_snapshot}
    if scope in ('supervised_primitive_task','supervised_hybrid_task'):
        from .primitive_service import PrimitiveService
        primitives = PrimitiveService(robot)
        methods.update({'primitive.observe': primitives.observe, 'primitive.prepare': primitives.prepare,
                        'primitive.execute': primitives.execute})
    if scope != 'supervised_primitive_task':
        methods.update({'env.preview': robot.preview, 'env.inspect': robot.inspect_action, 'env.step': robot.step,
                        'env.step_chunk': robot.step_chunk, 'env.preview_segment': robot.preview_segment,
                        'env.step_segment': robot.step_segment})
    if scope == 'supervised_hybrid_task':
        # serve() uses HTTPServer, so a tool call cannot overlap another tool call.
        # Invalidate on entry even if validation fails. A fresh observation is
        # required after changing backend; a primitive observation cannot be used
        # as an accidentally valid policy token created by its internal observe().
        def bind(name, method):
            def invoke(*args, **kwargs):
                if name.startswith('primitive.'):
                    robot.pending = None
                    try:
                        return method(*args, **kwargs)
                    finally:
                        robot.pending = None
                primitives.observation = None
                primitives.prepared = None
                return method(*args, **kwargs)
            return invoke
        methods = {name: (method if name == 'healthz' else bind(name,method))
                   for name,method in methods.items()}
    return methods


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--arm-motion", action="store_true")
    parser.add_argument("--port", type=int, default=9102)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    validate_config(cfg, args.arm_motion)
    import rospy
    rospy.init_node("cix_baxter_bridge", disable_signals=True)
    robot = Baxter(cfg, args.arm_motion)
    # Normal service restarts must release the USB streams, including SIGTERM.
    import signal
    def terminate(signum, frame):
        raise SystemExit(0)
    signal.signal(signal.SIGTERM, terminate)
    methods = build_rpc_methods(robot)
    try:
        serve(args.port, methods)
    finally:
        try:
            robot.close()
        finally:
            rospy.signal_shutdown("bridge stopped")


if __name__ == "__main__":
    main()
