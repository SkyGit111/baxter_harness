"""Compare a Baxter pregrasp against the ROS 1 MoveIt plan service, without execution."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time
import xml.etree.ElementTree as ET

from .contract import vector
from .pi05_contract import quaternion
from .primitives import eef_from_tcp


def model_inventory(description, semantic=None):
    root = ET.fromstring(description)
    bodies = {}
    for link in root.findall('link'):
        shapes = [shape.tag for shape in link.findall('collision/geometry/*')]
        if shapes:
            bodies[link.attrib['name']] = shapes
    if not bodies:
        raise ValueError('URDF has no collision bodies')
    result = {
        'model': root.attrib['name'],
        'urdf_sha256': hashlib.sha256(description.encode()).hexdigest(),
        'collision_links': bodies,
        'shape_counts': dict(Counter(s for shapes in bodies.values() for s in shapes)),
        'configured_groups': None,
        'disabled_collision_pairs': None,
    }
    if semantic is not None:
        srdf = ET.fromstring(semantic)
        if srdf.attrib['name'] != result['model']:
            raise ValueError('URDF and SRDF model names differ')
        result.update(srdf_sha256=hashlib.sha256(semantic.encode()).hexdigest(),
                      configured_groups=[g.attrib['name'] for g in srdf.findall('group')],
                      disabled_collision_pairs=[dict(p.attrib) for p in srdf.findall('disable_collisions')])
    return result


def pose_request(arm, group, position, orientation, planner_id=''):
    """A base-frame EEF pose goal. All intermediate joint configurations are free."""
    x, y, z = vector(position, 3, 'base-frame EEF position')
    qx, qy, qz, qw = quaternion(orientation)
    link = arm + '_gripper'
    common = {'header': {'frame_id': 'base'}, 'link_name': link, 'weight': 1.0}
    centre = {'position': {'x': x, 'y': y, 'z': z}, 'orientation': {'w': 1.0}}
    sphere_type = 2  # shape_msgs/SolidPrimitive.SPHERE
    position_constraint = dict(common, constraint_region={
        'primitives': [{'type': sphere_type, 'dimensions': [0.005]}], 'primitive_poses': [centre]})
    orientation_constraint = dict(common, orientation={'x': qx, 'y': qy, 'z': qz, 'w': qw},
                                  absolute_x_axis_tolerance=0.05, absolute_y_axis_tolerance=0.05,
                                  absolute_z_axis_tolerance=0.05)
    return {
        'group_name': group, 'planner_id': planner_id, 'allowed_planning_time': 5.0,
        'num_planning_attempts': 3, 'max_velocity_scaling_factor': 0.2,
        'max_acceleration_scaling_factor': 0.2,
        'goal_constraints': [{'position_constraints': [position_constraint],
                              'orientation_constraints': [orientation_constraint]}],
    }


def joint_names(arm):
    return {arm + '_' + joint for joint in ('s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2')}


def trajectory_summary(response, arm):
    code = response.error_code.val
    report = {'moveit_error_code': code, 'planner_reported_s': response.planning_time}
    if code != 1:  # MoveItErrorCodes.SUCCESS
        return dict(report, status='planning_failed')
    trajectory = response.trajectory.joint_trajectory
    names = list(trajectory.joint_names)
    if len(names) != 7 or set(names) != joint_names(arm) or not trajectory.points:
        raise ValueError('MoveIt success response has an unexpected trajectory group or no points')
    previous = None
    travel = [0.] * len(names)
    last_time = -1.
    for point in trajectory.points:
        values = vector(list(point.positions), len(names), 'MoveIt trajectory positions')
        stamp = point.time_from_start.to_sec()
        if not math.isfinite(stamp) or stamp < 0 or stamp < last_time:
            raise ValueError('Invalid MoveIt trajectory timing')
        if previous is not None:
            travel = [distance + abs(b-a) for distance, a, b in zip(travel, previous, values)]
        previous = values
        last_time = stamp
    return dict(report, status='pregrasp_plan_found', samples=len(trajectory.points),
                trajectory_duration_s=last_time, joint_travel_rad=dict(zip(names, travel)))


def assess_ros(request, arm):
    import genpy.message
    import rospy
    from moveit_msgs.msg import PlanningSceneComponents
    from moveit_msgs.srv import (GetMotionPlan, GetMotionPlanRequest, GetPlanningScene, GetPlanningSceneRequest,
                                 GetStateValidity, GetStateValidityRequest)

    rospy.init_node('baxter_moveit_assessment', anonymous=True, disable_signals=True)
    inventory = model_inventory(rospy.get_param('/robot_description'), rospy.get_param('/robot_description_semantic'))
    result = {'model_inventory': inventory}
    if request['group_name'] not in inventory['configured_groups']:
        raise ValueError('Unknown group. Available SRDF groups: ' + ', '.join(inventory['configured_groups']))
    for service in ('/get_planning_scene', '/check_state_validity', '/plan_kinematic_path'):
        rospy.wait_for_service(service, timeout=5)
    scene_request = GetPlanningSceneRequest()
    fields = PlanningSceneComponents
    scene_request.components.components = (fields.SCENE_SETTINGS | fields.ROBOT_STATE | fields.WORLD_OBJECT_GEOMETRY
                                          | fields.ROBOT_STATE_ATTACHED_OBJECTS | fields.ALLOWED_COLLISION_MATRIX)
    scene = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)(scene_request).scene
    if scene.robot_model_name != inventory['model']:
        raise ValueError('MoveIt scene uses a different robot model')
    joints = scene.robot_state.joint_state
    values = vector(list(joints.position), len(joints.name), 'planning scene state')
    if len(set(joints.name)) != len(joints.name) or not (joint_names('left') | joint_names('right')) <= set(joints.name):
        raise ValueError('Planning scene is missing complete, unambiguous dual arm state')
    # Store world geometry, including dimensions/poses, to verify the remeasured table.
    result['scene'] = {
        'name': scene.name, 'joint_stamp_s': joints.header.stamp.to_sec(),
        'snapshot_ros_time_s': rospy.Time.now().to_sec(),
        'start_joints_rad': dict(zip(joints.name, values)),
        'world_objects': [obj.id for obj in scene.world.collision_objects],
        'world_geometry': [genpy.message.strify_message(obj) for obj in scene.world.collision_objects],
        'attached_objects': [entry.object.id for entry in scene.robot_state.attached_collision_objects],
        'allowed_collision_matrix': genpy.message.strify_message(scene.allowed_collision_matrix),
    }
    validity_request = GetStateValidityRequest()
    # An empty group checks the entire starting configuration, including the inactive arm.
    validity_request.robot_state = scene.robot_state
    validity = rospy.ServiceProxy('/check_state_validity', GetStateValidity)(validity_request)
    result['start_state_valid'] = validity.valid
    result['start_contacts'] = [
        {'bodies': [contact.contact_body_1, contact.contact_body_2], 'depth_m': contact.depth}
        for contact in validity.contacts]
    if not validity.valid:
        return dict(result, status='invalid_start_state')
    query = GetMotionPlanRequest()
    genpy.message.fill_message_args(query.motion_plan_request, [request])
    query.motion_plan_request.start_state = scene.robot_state
    start = time.monotonic()
    response = rospy.ServiceProxy('/plan_kinematic_path', GetMotionPlan)(query).motion_plan_response
    result.update(service_wall_s=time.monotonic()-start, **trajectory_summary(response, arm))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--urdf', type=Path, help='Inspect saved robot_description without ROS')
    mode.add_argument('--ros', action='store_true', help='Query an existing Noetic move_group planning scene/service')
    parser.add_argument('--srdf', type=Path, help='Optional saved robot_description_semantic for offline inventory')
    parser.add_argument('--arm', choices=('left', 'right'))
    parser.add_argument('--group', help='SRDF planning group (default <arm>_arm)')
    parser.add_argument('--planner-id', default='', help='MoveIt planner configuration (default configured planner)')
    position = parser.add_mutually_exclusive_group()
    position.add_argument('--eef-base-m', type=float, nargs=3, metavar=('X', 'Y', 'Z'))
    position.add_argument('--tcp-base-m', type=float, nargs=3, metavar=('X', 'Y', 'Z'),
                          help='TCP position of a pregrasp move stage in base frame')
    parser.add_argument('--config', type=Path, help='Site profile for the TCP-to-gripper-link transform')
    parser.add_argument('--eef-xyzw', type=float, nargs=4, metavar=('X', 'Y', 'Z', 'W'))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args(argv)
    if args.srdf and args.ros:
        parser.error('--ros reads the SRDF from the ROS parameter server')
    if bool(args.tcp_base_m) != bool(args.config):
        parser.error('--tcp-base-m and --config must be used together')
    goal = (args.arm, args.eef_base_m or args.tcp_base_m, args.eef_xyzw)
    if (args.ros or any(goal) or args.group or args.planner_id) and not all(goal):
        parser.error('A pregrasp requires --arm, a base-frame position, and --eef-xyzw')
    report = {'applied': False, 'scope': 'pregrasp transit assessment', 'complete_task_checked': False}
    if all(goal):
        position = args.eef_base_m
        if args.config:
            raw = args.config.read_bytes()
            site = json.loads(raw.decode('utf-8-sig'))
            profile = site['arms'][args.arm] if 'arms' in site else site
            if profile.get('arm', args.arm) != args.arm:
                raise ValueError('Selected site profile is for the other arm')
            position = eef_from_tcp(args.tcp_base_m, args.eef_xyzw, profile['primitives'])
            report['tcp_input'] = {'arm': args.arm, 'tcp_base_m': args.tcp_base_m,
                                   'config_sha256': hashlib.sha256(raw).hexdigest()}
        report['request'] = pose_request(args.arm, args.group or args.arm+'_arm', position,
                                         args.eef_xyzw, args.planner_id)
    if args.urdf:
        report['model_inventory'] = model_inventory(args.urdf.read_text(encoding='utf-8-sig'),
                                                    args.srdf.read_text(encoding='utf-8-sig') if args.srdf else None)
        report['status'] = 'offline_inventory'
    else:
        report.update(assess_ros(report['request'], args.arm))
    encoded = json.dumps(report, ensure_ascii=False, allow_nan=False, indent=2)
    if args.output:
        with args.output.open('x', encoding='utf-8') as output:
            output.write(encoded + '\n')
        print(json.dumps({'status': report['status'], 'output': str(args.output)}))
    else:
        print(encoded)
    return 2 if report['status'] in ('planning_failed', 'invalid_start_state') else 0


if __name__ == '__main__':
    raise SystemExit(main())
