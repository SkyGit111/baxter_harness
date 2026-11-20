"""Archive a MoveIt 1 scene or check full-body paths against its immutable copy."""
import argparse
import hashlib
import io
import json
import math
import subprocess
import time
from pathlib import Path

from .collision import ArmCollisionError
from .moveit_guard import scene_box, segment_samples
from .planning import stage_phase
from .reachability import fingerprint


ARMS = ('left', 'right')
JOINTS = [arm + '_' + joint for arm in ARMS for joint in ('s0', 's1', 'e0', 'e1', 'w0', 'w1', 'w2')]
SCENE_FILES = ('robot.urdf', 'robot.srdf', 'planning.scene')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def capture(config, output):
    """Read ROS parameters and the scene service; add configured boxes to the private message."""
    import rospy
    from moveit_msgs.msg import PlanningSceneComponents as C
    from moveit_msgs.srv import GetPlanningScene, GetPlanningSceneRequest

    rospy.init_node('reachability_scene_reader', anonymous=True, disable_signals=True)
    rospy.wait_for_service('/get_planning_scene', timeout=5.)
    request = GetPlanningSceneRequest()
    request.components.components = (C.SCENE_SETTINGS | C.ROBOT_STATE | C.ROBOT_STATE_ATTACHED_OBJECTS |
                                     C.WORLD_OBJECT_NAMES | C.WORLD_OBJECT_GEOMETRY | C.OCTOMAP |
                                     C.TRANSFORMS | C.ALLOWED_COLLISION_MATRIX | C.LINK_PADDING_AND_SCALING)
    scene = rospy.ServiceProxy('/get_planning_scene', GetPlanningScene)(request).scene
    if scene.robot_model_name != 'baxter':
        raise ValueError('Expected a complete Baxter MoveIt scene')
    joint = scene.robot_state.joint_state
    joint_positions = dict(zip(joint.name, joint.position))
    if (len(joint.name) != len(joint.position) or len(joint.name) != len(joint_positions) or
            not (set(JOINTS) | {'head_pan'}) <= set(joint_positions)):
        raise ValueError('Missing or duplicated joints in MoveIt snapshot')
    boxes = {}
    required = set()
    for profile in config['arms'].values():
        planning = profile.get('planning', {})
        required.update(planning.get('moveit_required_world_objects', []))
        for box in planning.get('moveit_scene_boxes', []):
            if box['id'] in boxes and boxes[box['id']] != box:
                raise ValueError('Conflicting geometry between arm profiles: ' + box['id'])
            boxes[box['id']] = box
    world = {obj.id for obj in scene.world.collision_objects}
    attached = {obj.object.id for obj in scene.robot_state.attached_collision_objects}
    if boxes.keys() & (world | attached):
        raise ValueError('Site geometry shadows an existing MoveIt object')
    for box in boxes.values():
        scene.robot_state.attached_collision_objects.append(scene_box(box))
    if required - world - {box['id'] for box in boxes.values() if box['frame'] == 'base'}:
        raise ValueError('Required world geometry missing from snapshot')
    scene.is_diff = False
    scene.robot_state.is_diff = False
    wire = io.BytesIO()
    scene.serialize(wire)
    urdf = rospy.get_param('/robot_description')
    srdf = rospy.get_param('/robot_description_semantic')
    output.mkdir(parents=True, exist_ok=False)
    (output/'robot.urdf').write_bytes(urdf.encode('utf-8'))
    (output/'robot.srdf').write_bytes(srdf.encode('utf-8'))
    (output/'planning.scene').write_bytes(wire.getvalue())
    manifest = {'format': 'baxter-offline-moveit-v1', 'capture_unix_s': time.time(),
                'config_sha256': fingerprint(config), 'joint_positions': joint_positions,
                'configured_boxes': sorted(boxes), 'world_objects': sorted(world),
                'attachments': {obj.object.id: obj.link_name for obj in scene.robot_state.attached_collision_objects},
                'sha256': {name: digest(output/name) for name in SCENE_FILES}}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    return manifest


class ArchivedMoveItScene:
    def __init__(self, folder, checker, robot_description, config, states):
        manifest = json.loads((folder/'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('format') != 'baxter-offline-moveit-v1' or manifest['config_sha256'] != fingerprint(config):
            raise ValueError('MoveIt archive and site configuration differ')
        for name in SCENE_FILES:
            if digest(folder/name) != manifest['sha256'][name]:
                raise ValueError('MoveIt archive digest mismatch: ' + name)
        if (folder/'robot.urdf').read_text(encoding='utf-8') != robot_description:
            raise ValueError('KDL and archived MoveIt URDF differ')
        self.command = [str(checker.resolve())] + [str((folder/name).resolve()) for name in SCENE_FILES]
        self.start = {name: value for state in states.values() for name, value in zip(state['joint_names'], state['joint_rad'])}
        recorded = manifest['joint_positions']
        if (set(self.start) != set(JOINTS) or not all(math.isfinite(recorded[n]) and math.isfinite(self.start[n])
                                                     and abs(recorded[n] - self.start[n]) <= .015 for n in JOINTS)):
            raise ValueError('MoveIt scene and dual-arm starting snapshot differ by more than 0.015 rad')
        self.metadata = {'scene_sha256': manifest['sha256']['planning.scene'], 'srdf_sha256': manifest['sha256']['robot.srdf'],
                         'checker_sha256': digest(checker), 'world_objects': manifest['world_objects'],
                         'attachments': manifest['attachments'], 'capture_unix_s': manifest['capture_unix_s'],
                         'max_curve_step_rad': .01}
        status = self._validity([self.start])[0]
        if status != 'clear':
            raise ValueError('Archived full-body start is invalid: ' + status)

    def _validity(self, configurations):
        payload = ''.join(' '.join(format(row[n], '.16g') for n in JOINTS) + '\n' for row in configurations)
        process = subprocess.run(self.command, input=payload, capture_output=True, text=True, timeout=60)
        if process.returncode:
            raise RuntimeError('Offline MoveIt checker failed: {} {}'.format(process.returncode, process.stderr[-400:]))
        statuses = [line[len('BAXTER_FCL '):] for line in process.stdout.splitlines() if line.startswith('BAXTER_FCL ')]
        if len(statuses) != len(configurations) or any(s not in ('clear', 'joint_bounds') and not s.startswith('collision') for s in statuses):
            raise RuntimeError('Incomplete offline MoveIt validity evidence')
        return statuses

    def check(self, profile, stages, paths, initial, skill):
        rows = []
        indices = []
        for index, path in enumerate(paths):
            if path:
                for moving in segment_samples(initial, path, profile, stage_phase(stages, index, skill)):
                    rows.append(dict(self.start, **moving))
                    indices.append(index)
                initial = path[-1][0]
        if not rows:
            raise RuntimeError('Offline MoveIt received an empty motion path')
        for sample, status in enumerate(self._validity(rows)):
            if status != 'clear':
                raise ArmCollisionError('Archived MoveIt {} stage {} sample {}: {}'.format(skill, indices[sample], sample, status))
        return {'whole_body': True, 'checked_curve_samples': len(rows), 'max_curve_step_rad': .01}


def main():
    parser = argparse.ArgumentParser(description='Save the current planning scene for offline reachability checks')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New private directory')
    args = parser.parse_args()
    capture(json.loads(args.config.read_text(encoding='utf-8-sig')), args.output)
    print(args.output/'manifest.json')


if __name__ == '__main__':
    main()
