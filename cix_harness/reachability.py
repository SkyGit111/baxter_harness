"""Read-only, task-conditioned XYZ reachability sampling.

Passing samples are evidence for a complete planned pick and placement from one
recorded start state. They are NOT an actuator permission, a continuous volume,
or proof that failed samples are physically unreachable. No motor API is used.
"""
import argparse
import copy
import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS

from .contract import vector
from .pi05_contract import quaternion


def fingerprint(value):
    raw=value if isinstance(value,str) else json.dumps(value,sort_keys=True,allow_nan=False,separators=(',',':'))
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()


def yaw_orientation(orientation,degrees):
    """Rotate a grasp about base +Z; preserve the measured tool offset separately."""
    x,y,z,w=quaternion(orientation)
    angle=float(degrees)*math.pi/360.
    if not math.isfinite(angle):raise ValueError('Nonfinite grasp yaw')
    s,c=math.sin(angle),math.cos(angle)
    return quaternion([c*x-s*y,c*y+s*x,c*z+s*w,c*w-s*z])


def _axes(grid):
    result={}
    for axis in ('x','y','z'):
        values=grid.get(axis)
        if not isinstance(values,list) or not 1<=len(values)<=100:
            raise ValueError('Each grid axis needs 1..100 explicit sample coordinates')
        checked=vector(values,len(values),'grid '+axis)
        if len(set(checked))!=len(checked):raise ValueError('Duplicate grid coordinate')
        result[axis]=sorted(checked)
    if math.prod(map(len,result.values()))>10000:raise ValueError('Grid exceeds 10000 samples per arm')
    return result


def scan(config,states,robot_description,grid,destinations,evaluator,*,
         placement_skill='stack',yaw_degrees=(0,90,180,270),arms=('left','right')):
    """Evaluate each orientation through pick AND place/stack, not independent IK.

    evaluator(arm, point, destination, orientation, placement_skill) returns a
    receipt with checked_skills and complete_path_checked. Errors are recorded
    per candidate so an incomplete/failed check cannot become a passing sample.
    """
    if placement_skill not in ('place','stack'):raise ValueError('Placement must be place or stack')
    axes=_axes(grid)
    if not arms or len(set(arms))!=len(arms) or any(a not in ('left','right') for a in arms):
        raise ValueError('Select unique left/right arms')
    if not 1<=len(yaw_degrees)<=16:raise ValueError('Expected 1..16 grasp yaw candidates')
    yaws=vector(list(yaw_degrees),len(yaw_degrees),'grasp yaw degrees')
    report={'format':'task-reachability-v1','applied':False,'certified':False,
            'status':'offline_sampled_candidates','frame':'base',
            'sample_semantics':'object top reference point; z is base-frame height, not TCP or lift height',
            'task':['pick',placement_skill],'grid':axes,'yaw_degrees':yaws,
            'fingerprints':{'urdf_sha256':fingerprint(robot_description),
                            'config_sha256':fingerprint(config),'start_states_sha256':fingerprint(states)},
            'limitations':['No motor execution or grasp success verification.',
                           'Passing samples do not certify cells between them or their bounding box.',
                           'Results depend on start posture, grasp orientation, object geometry and destination.',
                           'Rejected candidates are not proof that every IK branch/path is impossible.',
                           'Geometry or calibration changes invalidate these results.',
                           'Head and cross-arm checks do not certify full torso/same-arm mesh or scene collision freedom.'],
            'arms':{}}
    effective=getattr(evaluator,'effective_config',None)
    if effective is not None:report['fingerprints']['effective_evaluator_config_sha256']=fingerprint(effective)
    started=time.monotonic()
    for arm in arms:
        destination=vector(destinations[arm],3,'placement target')
        original=quaternion(config['arms'][arm]['primitives']['tool_orientation_xyzw'])
        if arm not in states:raise ValueError('Missing recorded '+arm+' start state')
        samples=[];counts=Counter();slices=[]
        for z in axes['z']:
            layer=[]
            for y in axes['y']:
                for x in axes['x']:
                    point=[x,y,z];attempts=[]
                    for yaw in yaws:
                        q=yaw_orientation(original,yaw);began=time.monotonic()
                        attempt={'yaw_degrees':yaw,'orientation_xyzw':q}
                        try:
                            receipt=evaluator(arm,point,destination,q,placement_skill)
                            if (not isinstance(receipt,dict) or receipt.get('checked_skills')!=['pick',placement_skill]
                                    or receipt.get('complete_path_checked') is not True):
                                raise ValueError('Evaluator did not confirm the entire requested operation')
                            # Also reject nonfinite/nonserializable evidence immediately.
                            json.dumps(receipt,allow_nan=False)
                            attempt.update(status='candidate',evidence=receipt)
                        except Exception as error:
                            attempt.update(status='rejected',failure_type=type(error).__name__,reason=str(error))
                            counts[type(error).__name__]+=1
                        attempt['seconds']=round(time.monotonic()-began,6);attempts.append(attempt)
                    passed=any(a['status']=='candidate' for a in attempts)
                    sample={'point_base_m':point,'status':'candidate' if passed else 'no_candidate_found',
                            'attempts':attempts}
                    samples.append(sample);layer.append(len(samples)-1)
            slices.append({'z_base_m':z,'sample_indices':layer,
                           'candidate_count':sum(samples[i]['status']=='candidate' for i in layer),
                           'sample_count':len(layer)})
        good=[s for s in samples if s['status']=='candidate']
        report['arms'][arm]={'destination_base_m':destination,'start_state':copy.deepcopy(states[arm]),
                             'samples':samples,'z_slices':slices,'failure_counts':dict(counts),
                             'candidate_count':len(good),'sample_count':len(samples),
                             'sampled_candidate_z_values_m':sorted({s['point_base_m'][2] for s in good}),
                             'continuous_volume_certified':False}
    report['seconds']=round(time.monotonic()-started,6)
    return report


def snapshot_evaluator(config,states,robot_description):
    """Construct ROS/KDL types from saved data; never create robot interfaces.

    Requires ROS Python packages for pose types/interpolation, but no ROS master,
    subscribers, publishers or services. URDF and states come from explicit files.
    """
    from .collision import KDLArmCapsules,InterArmGuard,ArmCollisionError
    from .kinematics import ContinuousIK
    from .primitive_service import PrimitiveService,IKPreflightError
    from .planning import candidate_profiles
    from .primitives import plan,check_tcp,tcp_from_eef
    from genpy import Time
    config=copy.deepcopy(config)
    config['inter_arm_clearance']['check_head']=True
    solvers={a:ContinuousIK(a,robot_description=robot_description) for a in ('left','right')}
    models={a:KDLArmCapsules(a,config['inter_arm_clearance'],robot_description=robot_description)
            for a in ('left','right')}
    guard=InterArmGuard(config['inter_arm_clearance'],models)

    def pose(state):
        p=state['eef_m'];q=state['eef_xyzw']
        return {'position':NS(x=p[0],y=p[1],z=p[2]),'orientation':NS(x=q[0],y=q[1],z=q[2],w=q[3])}

    class RobotSnapshot:
        def __init__(self,arm,state,profile):
            self.state=state;self.cfg=profile;self.continuous_ik=solvers[arm]
            self.ros=NS(Time=NS(now=lambda:Time(0)))
            self.limb=NS(endpoint_pose=lambda:pose(state))
        def joints(self):return dict(zip(self.state['joint_names'],self.state['joint_rad']))
        def inside(self,xyz):
            if any(not lo<=v<=hi for lo,v,hi in zip(self.cfg['workspace_min_m'],xyz,self.cfg['workspace_max_m'])):
                raise ValueError('EEF outside configured workspace')
        def guard_pose(self,value):
            p,q=value['position'],value['orientation'];profile=self.cfg['primitives']
            check_tcp(tcp_from_eef([p.x,p.y,p.z],[q.x,q.y,q.z,q.w],profile),profile)

    def evaluate(arm,point,destination,orientation,placement_skill):
        state=copy.deepcopy(states[arm]);profile=copy.deepcopy(config['arms'][arm])
        profile['primitives']['tool_orientation_xyzw']=orientation
        if state.get('gripping') is not False:raise ValueError('Recorded task start must have an empty gripper')
        peer=states['right' if arm=='left' else 'left'];receipts=[]
        for skill,target in (('pick',point),(placement_skill,destination)):
            service=PrimitiveService(RobotSnapshot(arm,state,profile))
            try:
                # The outer scan already chose a grasp orientation. Match the
                # online direct-then-staged candidates for that orientation.
                for _,candidate in candidate_profiles(profile['primitives'],state,skill,{}):
                    stages=plan(skill,state,candidate,target)
                    try:
                        initial,paths=service._preflight(stages,{'state':state},time.monotonic(),skill=skill)
                        clearance=guard.path_clearance(arm,initial,paths,peer,peer)
                        break
                    except (IKPreflightError,ArmCollisionError) as error:
                        last_failure=error
                else:
                    raise last_failure
            except (ValueError,RuntimeError) as error:
                raise ValueError('{} failed ({}): {}'.format(skill,type(error).__name__,error)) from error
            heights=[tcp_from_eef(state['eef_m'],state['eef_xyzw'],profile['primitives'])[2]]
            for path in paths:
                for _,point_pose in path or []:
                    p,q=point_pose.pose.position,point_pose.pose.orientation
                    heights.append(tcp_from_eef([p.x,p.y,p.z],[q.x,q.y,q.z,q.w],profile['primitives'])[2])
            receipts.append({'skill':skill,'approach_path':candidate.get('approach_path','staged'),
                             'stages':len(stages),'ik_waypoints':sum(len(p) for p in paths if p),
                             'sampled_path_tcp_z_extrema_m':[min(heights),max(heights)],
                             'inter_arm_clearance':clearance})
            for stage,path in zip(stages,paths):
                if path:
                    joints,target_pose=path[-1];p=target_pose.pose.position;q=target_pose.pose.orientation
                    state.update(joint_names=list(joints),joint_rad=list(joints.values()),
                                 eef_m=[p.x,p.y,p.z],eef_xyzw=[q.x,q.y,q.z,q.w])
                elif stage['kind']=='gripper':state.update(gripping=not stage['open'],gripper_percent=100. if stage['open'] else 20.)
                elif stage['kind']=='verify_grasp':state['gripping']=stage['held']
        return {'checked_skills':['pick',placement_skill],'complete_path_checked':True,
                'hypothetical_grasp_success':True,'operations':receipts,
                'solver_scope':'Continuous local IK from the recorded posture; finite grasp yaw candidates, not global completeness.',
                'limitations':'Grasp transitions are simulated, not measured. Head and cross-arm checked; torso/same-arm mesh and scene obstacles not fully checked.'}
    evaluate.effective_config=config
    return evaluate


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config',required=True,type=Path)
    p.add_argument('--states',required=True,type=Path,help='Recorded dual observation, outcome or plain arm-state mapping')
    p.add_argument('--urdf',required=True,type=Path)
    p.add_argument('--grid',required=True,type=Path,help='JSON with explicit x, y, z arrays in base metres')
    p.add_argument('--destinations',required=True,type=Path,help='JSON mapping left/right to support top XYZ')
    p.add_argument('--arms',nargs='+',choices=['left','right'],default=['left','right'])
    p.add_argument('--yaw-degrees',nargs='+',type=float,default=[0,90,180,270])
    p.add_argument('--placement-skill',choices=['place','stack'],default='stack')
    p.add_argument('--output',required=True,type=Path)
    a=p.parse_args(argv)
    read=lambda path:json.loads(path.read_text(encoding='utf-8-sig'))
    cfg=read(a.config);raw=read(a.states);states=raw.get('arm_states',raw.get('after_arm_states',raw))
    description=a.urdf.read_text(encoding='utf-8-sig')
    evaluator=snapshot_evaluator(cfg,states,description)
    report=scan(cfg,states,description,read(a.grid),read(a.destinations),evaluator,
                arms=a.arms,yaw_degrees=a.yaw_degrees,placement_skill=a.placement_skill)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'output':str(a.output),'certified':False,
                      'arms':{arm:{k:v[k] for k in ('candidate_count','sample_count','sampled_candidate_z_values_m')}
                              for arm,v in report['arms'].items()}},ensure_ascii=False))


if __name__=='__main__':main()
