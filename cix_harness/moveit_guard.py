"""Full-body MoveIt state validity along the primitive executor's own joint path."""
import hashlib
import io
import math
import time

from .contract import vector
from .moveit_probe import joint_names
from .trajectory import JointTrajectory, motion_limits


class MoveItCollisionError(ValueError):
    """The scene was available and a sampled robot configuration was invalid."""


def message_digest(message):
    buf = io.BytesIO()
    message.serialize(buf)
    return hashlib.sha256(buf.getvalue()).hexdigest()


def options(planning):
    if not isinstance(planning, dict):
        raise ValueError('Expected a primitive planning configuration')
    backend = planning.get('collision_backend', 'builtin')
    required = planning.get('moveit_required_world_objects', [])
    if backend not in ('builtin', 'moveit'):
        raise ValueError('Unsupported primitive collision_backend')
    if (not isinstance(required, list) or any(not isinstance(v, str) or not v.strip() for v in required)
            or len(set(required)) != len(required) or (required and backend != 'moveit')):
        raise ValueError('Invalid moveit_required_world_objects')
    configured = planning.get('moveit_scene_boxes', [])
    if not isinstance(configured,list) or (configured and backend!='moveit'):
        raise ValueError('Configured scene boxes require the MoveIt backend')
    seen=set()
    for box in configured:
        if (not isinstance(box,dict) or not isinstance(box.get('id'),str) or not box['id'].strip()
                or box['id'] in seen or box.get('frame') not in ('base','left_gripper','right_gripper')
                or not isinstance(box.get('source'),str) or not box['source'].strip()):
            raise ValueError('Scene box requires a unique ID, a supported frame and measurement source')
        seen.add(box['id'])
        vector(box.get('xyz'),3,'collision box centre')
        orientation=vector(box.get('xyzw'),4,'collision box orientation')
        if abs(math.sqrt(sum(v*v for v in orientation))-1)>.0001 or min(vector(box.get('size_m'),3,'collision box size'))<=0:
            raise ValueError('Invalid collision box dimensions or rotation')
    return backend, sorted(required)


def scene_box(box):
    from geometry_msgs.msg import Pose
    from moveit_msgs.msg import AttachedCollisionObject
    from shape_msgs.msg import SolidPrimitive
    entry=AttachedCollisionObject()
    entry.link_name=entry.object.header.frame_id=box['frame']
    entry.object.id=box['id']
    centre=Pose()
    centre.position.x,centre.position.y,centre.position.z=box['xyz']
    centre.orientation.x,centre.orientation.y,centre.orientation.z,centre.orientation.w=box['xyzw']
    entry.object.primitives=[SolidPrimitive(type=SolidPrimitive.BOX,dimensions=box['size_m'])]
    entry.object.primitive_poses=[centre]
    entry.touch_links=[entry.link_name]
    if entry.link_name!='base':
        arm=entry.link_name.split('_')[0]
        entry.touch_links.extend([arm+'_gripper_base',arm+'_hand',arm+'_wrist'])
        entry.touch_links.extend(arm[0]+'_gripper_'+side+'_finger'+tip for side in ('l','r') for tip in ('','_tip'))
    return entry


def segment_samples(initial, path, cfg, phase):
    """Sample the spatial curve, irrespective of the feedback-driven time scale."""
    if not path:
        yield initial
        return
    knots = [q for q, _ in path]
    step = .01
    if cfg.get('planning', {}).get('trajectory_mode') == 'continuous':
        speed, acceleration = motion_limits(cfg, phase)
        yield from JointTrajectory(initial, knots, speed, acceleration * .5).sampled_positions(step)
        return
    yield initial
    for q in knots:
        if set(q) != set(initial):
            raise ValueError('Incomplete primitive joint knot')
        # The legacy quintic applies the same scalar fraction to all joints.
        count = max(1, math.ceil(max(abs(q[n] - initial[n]) for n in initial) / step))
        for i in range(1, count + 1):
            yield {n: initial[n] + (q[n] - initial[n]) * i / count for n in initial}
        initial = q


class MoveItGuard:
    def __init__(self, robot, required_world):
        self.robot = robot
        self.arm = robot.cfg['arm']
        self.active = joint_names(self.arm)
        self.inactive = joint_names('right' if self.arm == 'left' else 'left') | {'head_pan'}
        self.required_world = required_world
        self.execution_reference = None
        self.ros = None

    def _connect(self):
        from moveit_msgs.srv import GetPlanningScene, GetStateValidity
        import rospy
        for service in ('/get_planning_scene', '/check_state_validity'):
            rospy.wait_for_service(service, timeout=3.)
        self.scene_service = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)
        self.validity = rospy.ServiceProxy('/check_state_validity', GetStateValidity, persistent=True)
        self.ros = rospy

    def _scene(self):
        from moveit_msgs.msg import PlanningSceneComponents as C
        from moveit_msgs.srv import GetPlanningSceneRequest
        req = GetPlanningSceneRequest()
        req.components.components = (C.SCENE_SETTINGS | C.ROBOT_STATE | C.ROBOT_STATE_ATTACHED_OBJECTS
                                     | C.WORLD_OBJECT_GEOMETRY | C.ALLOWED_COLLISION_MATRIX)
        scene = self.scene_service(req).scene
        joints = scene.robot_state.joint_state
        names = list(joints.name)
        if (scene.robot_model_name != 'baxter' or len(names) != len(set(names))
                or len(names) != len(joints.position) or not (self.active | self.inactive) <= set(names)
                or any(not math.isfinite(x) for x in joints.position)):
            raise RuntimeError('MoveIt requires the complete Baxter scene and both arms plus head')
        world = {}
        for obj in scene.world.collision_objects:
            shapes = len(obj.primitives) + len(obj.meshes) + len(obj.planes)
            if not obj.id or obj.id in world or shapes < 1:
                raise RuntimeError('Missing or ambiguous MoveIt world collision geometry')
            # Normalize the transport timestamp without modifying any remote scene.
            obj.header.stamp = self.ros.Time(0)
            world[obj.id] = {'frame': obj.header.frame_id, 'shape_count': shapes,
                             'geometry_sha256': message_digest(obj),
                             'primitive_dimensions': [list(p.dimensions) for p in obj.primitives]}
        state = scene.robot_state
        state.is_diff = False
        configured=self.robot.cfg.get('planning',{}).get('moveit_scene_boxes',[])
        occupied=set(world)|{item.object.id for item in state.attached_collision_objects}
        for box in configured:
            if box['id'] in occupied:
                raise RuntimeError('Configured MoveIt geometry shadows a live object: '+box['id'])
            entry=scene_box(box)
            state.attached_collision_objects.append(entry)
            occupied.add(box['id'])
            if box['frame']=='base':
                world[box['id']]={'frame':'base','shape_count':1,'geometry_sha256':message_digest(entry.object),
                                  'primitive_dimensions':[list(box['size_m'])],'source':box['source'],
                                  'scope':'request_fixed_to_base'}
        missing = set(self.required_world) - world.keys()
        if missing:
            raise RuntimeError('MoveIt world lacks required collision geometry: ' + ', '.join(sorted(missing)))
        attachments = {}
        for entry in state.attached_collision_objects:
            # Attachment pose is expressed in its link frame, independent of the
            # arm angles in this specific robot state.
            entry.object.header.stamp = self.ros.Time(0)
            attachments[entry.object.id] = {'link': entry.link_name, 'touch_links': list(entry.touch_links),
                                            'geometry_sha256': message_digest(entry.object)}
        return state, {'world_objects': world, 'required_world_objects_present': list(self.required_world),
                       'allowed_collision_matrix_sha256': message_digest(scene.allowed_collision_matrix),
                       'attached_objects': attachments,
                       'configured_box_sources':{box['id']:box['source'] for box in configured},
                       'world_geometry_physically_verified_by_this_check': False}

    def peer(self):
        stamp, values = self.robot.full_body_joint_sample()
        if not 0 <= time.monotonic() - stamp < .5 or not (self.active | self.inactive) <= values.keys():
            raise RuntimeError('Fresh measured joints of both arms and head required by MoveIt')
        return values

    def check_inactive(self, reference):
        values = self.peer()
        if any(abs(values[n] - reference[n]) > .015 for n in self.inactive):
            raise RuntimeError('Inactive arm or head moved during MoveIt collision validation')
        return values

    def _valid(self, state):
        from moveit_msgs.srv import GetStateValidityRequest
        req = GetStateValidityRequest()
        req.group_name = ''  # Whole robot, including its inactive arm and head.
        req.robot_state = state
        return self.validity(req)

    def check(self, initial, segments, *, executing=False):
        began = time.monotonic()
        if set(initial) != self.active:
            raise RuntimeError('MoveIt requires all seven selected arm joints')
        reference = (self.check_inactive(self.execution_reference) if self.execution_reference is not None
                     else self.peer())
        try:
            if self.ros is None:
                self._connect()
            state, evidence = self._scene()
            joints = state.joint_state
            passive = dict(zip(joints.name, joints.position))
            samples = 0
            for stage_index, phase, path in segments:
                for q in segment_samples(initial, path, self.robot.cfg, phase):
                    live = self.check_inactive(reference)
                    joints.position = [q[n] if n in q else live.get(n, passive[n]) for n in joints.name]
                    response = self._valid(state)
                    samples += 1
                    if response.valid is not True:
                        contacts = [(c.contact_body_1, c.contact_body_2) for c in response.contacts]
                        raise MoveItCollisionError(
                            f'MoveIt rejected stage {stage_index}, sample {samples}, contacts {contacts}')
                if path:
                    initial = path[-1][0]
            if samples == 0:
                raise ValueError('MoveIt collision check received no stages')
            self.check_inactive(reference)
            _, final_scene = self._scene()
            if final_scene != evidence:
                raise RuntimeError('MoveIt scene geometry or allowed collision matrix changed during validation')
            if executing:
                # Monitor from the pose whose full geometry just cleared this
                # segment. Never move the anchor when a check fails.
                self.execution_reference = reference
        except MoveItCollisionError:
            raise
        except Exception:
            # No fallback. Reconnect cleanly on a later explicit request.
            if self.ros is not None:
                self.validity.close()
            self.ros = None
            raise
        return {**evidence, 'whole_robot': True, 'samples': samples, 'max_joint_step_rad': .01,
                'peer_source': '/robot/joint_states', 'seconds': time.monotonic() - began}
