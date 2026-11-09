"""Robot-side primitive RPCs; one-use plans, full IK preflight, measured outcomes."""
import math
import time
import uuid
from copy import deepcopy
from .primitives import validate_profile, plan, ground_pixel, eef_from_tcp

class PrimitiveService:
    def __init__(self,robot):
        self.robot=robot
        self.profile=validate_profile(robot.cfg['primitives'])
        self.observation=None
        self.prepared=None

    def observe(self):
        self.prepared=None
        obs=self.robot.observe()
        depth=self.robot.camera.last_depth
        if depth is None:raise RuntimeError('Aligned depth missing')
        self.observation=(obs,time.monotonic(),depth.copy())
        obs['primitive_skills']=['pick','place','stack','move_to','lift','retreat','open','close','home']
        return obs

    def prepare(self,obs_id,skill,uv=None,target_base_m=None):
        from geometry_msgs.msg import PoseStamped
        from tf import transformations as tf
        from .contract import vector
        self.prepared=None
        observation,self.observation=self.observation,None
        if observation is None or observation[0]['id']!=obs_id:raise ValueError('Observation consumed or superseded')
        obs,began,depth=observation
        r=self.robot;p=self.profile
        if time.monotonic()-began>p['perception_max_age_s']:raise ValueError('Scene observation expired')
        if uv is not None and target_base_m is not None:raise ValueError('Choose one target representation')
        target=(ground_pixel(uv,depth,r.camera.intrinsics,p) if uv is not None else target_base_m)
        if target is not None:target=vector(target,3,'base target')
        stages=plan(skill,obs['state'],p,target)
        initial=r.joints()
        measured=dict(zip(obs['state']['joint_names'],obs['state']['joint_rad']))
        if set(initial)!=set(measured) or max(abs(initial[k]-measured[k]) for k in initial)>.015:
            raise ValueError('Arm moved during visual planning')
        start=r.limb.endpoint_pose();observed=start;seed=initial;paths=[]
        # Compute every leg before any gripper/motor operation. No prefix fallback.
        for stage in stages:
            if stage['kind']!='move':paths.append(None);continue
            dest=eef_from_tcp(stage['tcp_m'],stage['xyzw'],p)
            a=[observed['position'].x,observed['position'].y,observed['position'].z]
            oq=observed['orientation'];qa=[oq.x,oq.y,oq.z,oq.w];qb=stage['xyzw']
            distance=math.dist(a,dest)
            if distance>p['max_leg_m']:raise ValueError('Primitive leg exceeds configured travel')
            angle=2*math.acos(min(1,abs(sum(x*y for x,y in zip(qa,qb)))))
            count=max(1,math.ceil(distance/.01),math.ceil(angle/.015));path=[]
            for i in range(1,count+1):
                f=i/count;point=PoseStamped();point.header.frame_id='base';point.header.stamp=r.ros.Time.now()
                xyz=[x+f*(y-x) for x,y in zip(a,dest)]
                q=list(map(float,tf.quaternion_slerp(qa,qb,f)))
                r.inside(xyz)
                point.pose.position.x,point.pose.position.y,point.pose.position.z=xyz
                point.pose.orientation.x,point.pose.orientation.y,point.pose.orientation.z,point.pose.orientation.w=q
                pose={'position':point.pose.position,'orientation':point.pose.orientation}
                r.guard_pose(pose)
                from .primitives import tcp_from_eef
                tcp=tcp_from_eef(xyz,q,p)
                if tcp[2]<p['table_z_m']+p['table_clearance_m']:raise ValueError('Interpolated TCP crosses table')
                if any(not lo<=v<=hi for v,lo,hi in zip(tcp,p['tcp_workspace_min_m'],p['tcp_workspace_max_m'])):
                    raise ValueError('Interpolated TCP leaves workspace')
                solution=r.continuous_ik.solve(point,seed,observed)
                path.append((solution,point));seed=solution;observed=pose
            paths.append(path)
        operations=sum(s['kind'] in ('move','gripper') for s in stages)
        if r.motion_remaining is None or r.motion_remaining<operations:raise RuntimeError('Insufficient primitive stage budget')
        if time.monotonic()-began>p['perception_max_age_s']:raise ValueError('Scene expired during full IK preflight')
        if max(abs(r.joints()[k]-initial[k]) for k in initial)>.015:raise ValueError('Arm moved during IK preflight')
        token=uuid.uuid4().hex
        self.prepared=(token,time.monotonic(),initial,stages,paths,operations,skill,obs['state']['gripping'])
        return {'plan_id':token,'obs_id':obs_id,'skill':skill,'target_base_m':target,'stages':stages,
                'applied':False,'ik_preflight':True,'expires_in_s':r.cfg['max_observation_age_s']}

    def execute(self,plan_id):
        prepared,self.prepared=self.prepared,None
        if prepared is None or prepared[0]!=plan_id:raise ValueError('Plan consumed or superseded; never replay')
        token,created,initial,stages,paths,operations,skill,holding=prepared;r=self.robot;p=self.profile
        if not r.armed or r.cfg.get('commissioning_scope')!='supervised_primitive_task':raise RuntimeError('Primitive bridge is read-only or wrong scope')
        if p.get('calibrated') is not True:raise RuntimeError('Primitive geometry is not calibrated')
        if time.monotonic()-created>r.cfg['max_observation_age_s']:raise RuntimeError('Plan expired')
        if max(abs(r.joints()[k]-initial[k]) for k in initial)>.015:raise RuntimeError('Arm moved since planning')
        if r.motion_remaining<operations:raise RuntimeError('Stage budget exhausted')
        r.ready();r.motion_remaining-=operations;receipts=[]
        r.audit('primitive_dispatch',plan_id=token,skill=skill,stages=stages)
        try:
            for stage,path in zip(stages,paths):
                r.ready();r.joints()
                if stage['kind']=='move':
                    if holding and not r.grip.gripping():raise RuntimeError('Held object lost before movement')
                    r.move_segment(r.joints(),path,r.endpoint_position(),max_travel=p['max_leg_m'])
                    d,a=r.arrival_error(path[-1][1])
                    if d>r.cfg.get('arrival_position_tolerance_m',.01) or a>.04:
                        raise RuntimeError('Primitive stage did not reach target; do not proceed to contact/release')
                    if holding and not r.grip.gripping():raise RuntimeError('Held object lost during movement')
                elif stage['kind']=='gripper':
                    if not r.grip.command_position(100. if stage['open'] else 0.,block=True,timeout=2):
                        raise RuntimeError('Gripper command timeout')
                    time.sleep(.25)
                    if stage['open'] and r.grip.position()<70:raise RuntimeError('Gripper failed to open')
                    holding=bool(r.grip.gripping())
                else:
                    if bool(r.grip.gripping()) != stage['held']:raise RuntimeError('Empty grasp or release not confirmed')
                r.ready()
                receipt={'stage':stage,'eef_m':r.endpoint_position(),'gripping':bool(r.grip.gripping())}
                receipts.append(receipt);r.audit('primitive_stage_applied',plan_id=token,**receipt)
        except BaseException as error:
            try:r.limb.set_joint_positions(r.joints());r.grip.stop(block=False)
            except Exception as stop_error:r.ros.logwarn('Primitive stop failed: %s',stop_error)
            r.audit('primitive_failed',plan_id=token,error=str(error),completed_stages=len(receipts))
            raise
        result={'plan_id':token,'skill':skill,'applied':True,'completed_stages':receipts,
                'gripping':bool(r.grip.gripping()),'motion_remaining':r.motion_remaining}
        r.audit('primitive_completed',**result)
        return result
