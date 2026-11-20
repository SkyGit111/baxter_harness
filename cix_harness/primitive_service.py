"""Robot-side primitive RPCs; one-use plans, full IK preflight, measured outcomes."""
import math
import time
import uuid
from contextlib import contextmanager
from .kinematics import IKUnreachableError
from .primitives import SKILLS, TargetDepthError, TargetWorkspaceError, validate_profile, plan, ground_pixel, eef_from_tcp, tcp_from_eef, check_tcp
from .workspace import workspace_metadata
from .planning import candidate_profiles,stage_phase
from .collision import ArmCollisionError, PeerJointMovedError
from .moveit_guard import MoveItGuard, MoveItCollisionError, options as collision_options


class SceneChangedError(ValueError):
    pass


class IKPreflightError(ValueError):
    pass


class EmptyGrasp(RuntimeError):
    pass


def check_support_departure(start_tcp, end_tcp, start_q, end_q, profile):
    """A newly grasped object starts on its support, before free-space clearance.

    This exception is only for the first straight upward pick leg. It never
    permits sliding, lowering, rotating, or an endpoint that is still in contact.
    Runtime grasp verification must succeed before this leg is dispatched.
    """
    if math.dist(start_tcp[:2],end_tcp[:2])>1e-7 or end_tcp[2]<=start_tcp[2]:
        raise ValueError('Support departure must be straight upward')
    if abs(sum(a*b for a,b in zip(start_q,end_q)))<1-1e-9:
        raise ValueError('Support departure must preserve orientation')
    check_tcp(start_tcp,profile)
    check_tcp(end_tcp,profile,held_bottom_m=profile['held_bottom_below_tcp_m'])

@contextmanager
def _measure(timings, name, aggregate=None):
    """Count elapsed work even when it raises; nested totals are not additive."""
    started=time.monotonic()
    try:
        yield
    finally:
        seconds=time.monotonic()-started
        timings[name]=timings.get(name,0.)+seconds
        if aggregate is not None:
            aggregate[name]=aggregate.get(name,0.)+seconds


def _peer_path_check(robot,skill,holding):
    check=getattr(type(robot),'check_other_arm_path',None)
    if skill=='clear_view' and holding is False:
        return getattr(type(robot),'check_empty_clear_view_path',check)
    return check


def undispatched_peer_failure(plan_id,prepared,error,remaining,duration):
    if prepared is None or prepared[0]!=plan_id:
        raise ValueError('Plan consumed or superseded; never replay')
    return {'plan_id':plan_id,'skill':prepared[6],'status':'failed','failure_code':'execution_failed',
            'failure_type':type(error).__name__,'message':str(error),'applied':False,
            'completed_stages':[],'completed_stage_count':0,'confirmed_commands':0,
            'failed_stage':{'stage_index':0,'phase':'dispatch_validation','status':'failed',
                            'duration_s':duration,'command_state':'not_dispatched'},
            'outcome_uncertain':False,'recoverable':False,'retryable':False,'stop_attempted':False,
            'stopped':None,'stop_error':None,'gripping':None,'motion_remaining':remaining}


class PrimitiveService:
    def __init__(self,robot):
        self.robot=robot
        options=robot.cfg.get('planning',{})
        if not isinstance(options,dict) or options.get('trajectory_mode','legacy') not in ('legacy','continuous'):
            raise ValueError('Unsupported primitive trajectory mode')
        allowed=options.get('primitive_skills',list(SKILLS))
        if (not isinstance(allowed,list) or not allowed or any(s not in SKILLS for s in allowed)
                or len(allowed)!=len(set(allowed))):
            raise ValueError('Expected distinct supported primitive_skills')
        self.skills=tuple(allowed)
        backend, required_world = collision_options(options)
        self.moveit = MoveItGuard(robot, required_world) if backend == 'moveit' else None
        robot.primitive_moveit_guard = self.moveit
        self.profile=validate_profile(robot.cfg['primitives'])
        self.observation=None
        self.prepared=None

    def observe(self):
        started=time.monotonic();timings={}
        try:
            with _measure(timings,'robot_observe_total'):
                self.prepared=None
                self.observation=None
                observe=self.robot.observe
                if (self.robot.cfg.get('commissioning_scope')=='supervised_hybrid_task'
                        and callable(getattr(type(self.robot),'observe_for_primitives',None))):
                    observe=self.robot.observe_for_primitives
                obs=observe()
            # Preserve lower-level camera/state timings if the robot publishes them.
            timings={**obs.get('timings_s',{}),**timings}
            with _measure(timings,'observation_metadata'):
                self._describe_observation(obs)
        except BaseException as error:
            self.prepared=None;self.observation=None
            timings['observe_total']=time.monotonic()-started
            self.robot.audit('primitive_observe_failed',status='failed',error=str(error),timings_s=dict(timings))
            raise
        timings['observe_total']=time.monotonic()-started
        obs['timings_s']=timings
        self.robot.audit('primitive_observed',obs_id=obs['id'],timings_s=dict(timings))
        return obs

    def _describe_observation(self,obs):
        depth=self.robot.camera.last_depth
        if depth is None:raise RuntimeError('Aligned depth missing')
        self.observation=(obs,time.monotonic(),depth.copy())
        tcp=tcp_from_eef(obs['state']['eef_m'],obs['state']['eef_xyzw'],self.profile)
        views=self.profile.get('observation_tcp_m',[])
        obs['recovery']={'view_indices':[i for i,v in enumerate(views) if math.dist(tcp,v)>=.02],
                         'motion_remaining':self.robot.motion_remaining}
        obs['primitive_skills']=[s for s in self.skills
                                 if (s!='clear_view' or obs['recovery']['view_indices'])
                                 and (s!='home' or self.profile.get('home_tcp_m') is not None)]
        import pyrealsense2 as rs
        obs['workspace']=workspace_metadata(self.profile,self.robot.camera.intrinsics,rs.rs2_project_point_to_pixel)

    def prepare(self,obs_id,skill,uv=None,target_base_m=None,view_index=0):
        started=time.monotonic()
        timings={'grounding':0.,'skill_plan':0.,'ik_preflight':0.}
        try:
            result=self._prepare(obs_id,skill,uv,target_base_m,view_index,timings)
        except (TargetDepthError,TargetWorkspaceError,SceneChangedError,IKPreflightError,MoveItCollisionError) as error:
            # Only these explicit, pre-actuation failures permit replanning.
            code={TargetDepthError:'target_depth',TargetWorkspaceError:'target_workspace',
                  SceneChangedError:'scene_changed',IKPreflightError:'ik_preflight',
                  MoveItCollisionError:'moveit_collision'}[type(error)]
            result={'status':'rejected','obs_id':obs_id,'skill':skill,'applied':False,
                    'failure_code':code,'message':str(error),'recoverable':True}
        except BaseException as error:
            timings['prepare_total']=time.monotonic()-started
            self.robot.audit('primitive_prepare_failed',obs_id=obs_id,skill=skill,applied=False,
                             status='failed',error=str(error),timings_s=dict(timings))
            raise
        timings['prepare_total']=time.monotonic()-started
        result['timings_s']=timings
        self.robot.audit('primitive_prepared' if result['status']=='ready' else 'primitive_prepare_rejected',**result)
        return result

    def _prepare(self,obs_id,skill,uv=None,target_base_m=None,view_index=0,timings=None):
        from .contract import vector
        self.prepared=None
        if skill not in self.skills:
            raise ValueError('Primitive is outside the configured skill scope: '+str(skill))
        observation,self.observation=self.observation,None
        if observation is None or observation[0]['id']!=obs_id:raise ValueError('Observation consumed or superseded')
        obs,began,depth=observation
        r=self.robot;p=self.profile
        if time.monotonic()-began>p['perception_max_age_s']:raise SceneChangedError('Scene observation expired')
        if uv is not None and target_base_m is not None:raise ValueError('Choose one target representation')
        if skill not in ('pick','place','stack','move_to') and (uv is not None or target_base_m is not None):
            raise ValueError('This skill does not accept a target')
        if timings is None:timings={}
        with _measure(timings,'grounding'):
            target=(ground_pixel(uv,depth,r.camera.intrinsics,p) if uv is not None else target_base_m)
            if target is not None:target=vector(target,3,'base target')
        peer_check=_peer_path_check(r,skill,obs['state'].get('gripping'))
        attempts=[];last_error=None;moveit_receipt=None
        for yaw,candidate in candidate_profiles(p,obs['state'],skill,r.cfg.get('planning',{})):
            attempt_started=time.monotonic()
            route=candidate.get('approach_path','staged')
            motion_schedule=[]
            try:
                with _measure(timings,'skill_plan'):
                    stages=plan(skill,obs['state'],candidate,target,view_index)
                operations=sum(s['kind'] in ('move','gripper') for s in stages)
                if r.motion_remaining is None or r.motion_remaining<operations:
                    attempts.append({'yaw_rad':yaw,'approach_path':route,'status':'rejected',
                                     'reason':'stage_budget','required_stages':operations})
                    continue
                with _measure(timings,'ik_preflight'):
                    initial,paths=self._preflight(stages,obs,began,skill=skill,motion_schedule=motion_schedule)
                if callable(peer_check):
                    with _measure(timings,'other_arm_preflight'):
                        if peer_check(r,paths) is False:raise ValueError('Other arm path check rejected the plan')
                if self.moveit is not None:
                    if len(paths) != len(stages):
                        raise RuntimeError('Incomplete primitive paths for MoveIt')
                    with _measure(timings,'moveit_full_path'):
                        segments = [(i, stage_phase(stages, i, skill), path) for i, path in enumerate(paths)]
                        moveit_receipt = self.moveit.check(initial, segments)
            except (IKPreflightError,ArmCollisionError,MoveItCollisionError) as error:
                attempts.append({'yaw_rad':yaw,'approach_path':route,'status':'rejected','reason':str(error),'seconds':time.monotonic()-attempt_started})
                last_error=error
                continue
            attempts.append({'yaw_rad':yaw,'approach_path':route,'status':'ready','seconds':time.monotonic()-attempt_started})
            break
        else:
            r.audit('primitive_candidates_rejected',obs_id=obs_id,skill=skill,candidates=attempts)
            if last_error is not None:
                raise last_error
            return {'status':'rejected','obs_id':obs_id,'skill':skill,'applied':False,'recoverable':False,
                    'failure_code':'stage_budget','message':'Insufficient primitive stage budget',
                    'required_stages':min(item['required_stages'] for item in attempts),
                    'motion_remaining':r.motion_remaining,'planning_candidates':attempts}
        if time.monotonic()-began>p['perception_max_age_s']:raise SceneChangedError('Scene expired during path preflight')
        final=r.joints()
        if set(final)!=set(initial) or max(abs(final[k]-initial[k]) for k in initial)>.015:
            raise SceneChangedError('Arm moved during path preflight')
        token=uuid.uuid4().hex
        self.prepared=(token,time.monotonic(),initial,stages,paths,operations,skill,obs['state']['gripping'])
        return {'status':'ready','plan_id':token,'obs_id':obs_id,'skill':skill,'target_base_m':target,'stages':stages,
                'applied':False,'ik_preflight':True,'required_stages':operations,
                'expires_in_s':r.cfg['max_observation_age_s'],'planning_candidates':attempts,
                'selected_approach_path':route,'motion_schedule':motion_schedule,
                'collision_backend':'moveit' if self.moveit is not None else 'builtin',
                'moveit_full_path':moveit_receipt,
                'nominal_motion_seconds':sum(s['nominal_duration_s'] for s in motion_schedule) if motion_schedule else None,
                'selected_yaw_rad':yaw,'trajectory_mode':r.cfg.get('planning',{}).get('trajectory_mode','legacy')}

    def _preflight(self,stages,obs,began,skill=None,motion_schedule=None):
        from geometry_msgs.msg import PoseStamped
        from tf import transformations as tf
        from .primitives import tcp_from_eef
        r=self.robot;p=self.profile
        initial=r.joints()
        measured=dict(zip(obs['state']['joint_names'],obs['state']['joint_rad']))
        if set(initial)!=set(measured) or max(abs(initial[k]-measured[k]) for k in initial)>.015:
            raise SceneChangedError('Arm moved during visual planning')
        start=r.limb.endpoint_pose();observed=start;seed=initial;paths=[];holding=obs['state']['gripping']
        pending_departure=skill=='pick' and holding is False
        # Compute every leg before any gripper/motor operation. No prefix fallback.
        for stage_index,stage in enumerate(stages):
            if stage['kind']!='move':
                if stage['kind']=='gripper':holding=not stage['open']
                elif stage['kind']=='verify_grasp':holding=stage['held']
                paths.append(None);continue
            dest=eef_from_tcp(stage['tcp_m'],stage['xyzw'],p)
            a=[observed['position'].x,observed['position'].y,observed['position'].z]
            oq=observed['orientation'];qa=[oq.x,oq.y,oq.z,oq.w];qb=stage['xyzw']
            departing=pending_departure and holding
            if departing:
                check_support_departure(tcp_from_eef(a,qa,p),stage['tcp_m'],qa,qb,p)
                pending_departure=False
            curve_initial=dict(seed)
            distance=math.dist(a,dest)
            if distance>p['max_leg_m']+1e-9:raise ValueError('Primitive leg exceeds configured travel')
            angle=2*math.acos(min(1,abs(sum(x*y for x,y in zip(qa,qb)))))
            # On a combined transfer the physical TCP must follow the checked
            # line above the table, even as the offset tool turns about the EEF.
            tcp_start=None
            if stage.get('motion_phase')=='transit':
                tcp_start=tcp_from_eef(a,qa,p)
                distance=math.dist(tcp_start,stage['tcp_m'])+math.dist(p['tcp_offset_tool_m'],[0.,0.,0.])*angle
            count=max(1,math.ceil(distance/.01),math.ceil(angle/.015));path=[]
            for i in range(1,count+1):
                f=i/count;point=PoseStamped();point.header.frame_id='base';point.header.stamp=r.ros.Time.now()
                xyz=[x+f*(y-x) for x,y in zip(a,dest)]
                q=list(map(float,tf.quaternion_slerp(qa,qb,f)))
                if tcp_start is not None:
                    xyz=eef_from_tcp([x+f*(y-x) for x,y in zip(tcp_start,stage['tcp_m'])],q,p)
                r.inside(xyz)
                point.pose.position.x,point.pose.position.y,point.pose.position.z=xyz
                point.pose.orientation.x,point.pose.orientation.y,point.pose.orientation.z,point.pose.orientation.w=q
                pose={'position':point.pose.position,'orientation':point.pose.orientation}
                r.guard_pose(pose)
                tcp=tcp_from_eef(xyz,q,p)
                try:check_tcp(tcp,p,held_bottom_m=p['held_bottom_below_tcp_m'] if holding and not departing else 0.)
                except ValueError as error:
                    raise ValueError('Interpolated TCP leaves workspace or crosses table: '+str(error)) from error
                try:solution=r.continuous_ik.solve(point,seed,observed)
                except IKUnreachableError as error:raise IKPreflightError(str(error)) from error
                path.append((solution,point));seed=solution;observed=pose
            if r.cfg.get('planning',{}).get('trajectory_mode')=='continuous':
                phase=stage_phase(stages,stage_index,skill)
                meta=self._check_curve(curve_initial,path,holding,departing,phase)
                if motion_schedule is not None:
                    motion_schedule.append({'stage_index':stage_index,'phase':phase,
                                            'nominal_duration_s':meta['nominal_duration_s'],
                                            'joint_speed_limit_rad_s':meta['joint_speed_limit_rad_s'],
                                            'ik_waypoints':len(path)})
            paths.append(path)
        if time.monotonic()-began>p['perception_max_age_s']:raise SceneChangedError('Scene expired during full IK preflight')
        if max(abs(r.joints()[k]-initial[k]) for k in initial)>.015:raise SceneChangedError('Arm moved during IK preflight')
        return initial,paths

    def _check_curve(self,initial,path,holding,departing,phase):
        """Check the same curve built by the executor, including measured start drift."""
        from .trajectory import JointTrajectory,motion_limits
        r=self.robot;p=self.profile
        speed,acceleration=motion_limits(r.cfg,phase)
        trajectory=JointTrajectory(initial,[q for q,_ in path],speed,acceleration*.5)
        for joints in trajectory.sampled_positions(.005):
            pose=r.continuous_ik.forward(joints);pos,q=pose['position'],pose['orientation']
            xyz=[pos.x,pos.y,pos.z];r.inside(xyz);r.guard_pose(pose)
            tcp=tcp_from_eef(xyz,[q.x,q.y,q.z,q.w],p)
            try:check_tcp(tcp,p,held_bottom_m=p['held_bottom_below_tcp_m'] if holding and not departing else 0.)
            except ValueError as error:raise IKPreflightError('Continuous curve violates TCP clearance: '+str(error)) from error
        return trajectory.metadata

    def execute(self,plan_id):
        started=time.monotonic()
        timings={'execution_validation':0.,'stage_validation':0.,'motion':0.,'motion_verification':0.,
                 'gripper_command':0.,'gripper_settle':0.,'gripper_verification':0.,'grasp_verification':0.,'stop':0.}
        try:
            result=self._execute(plan_id,timings,started)
        except BaseException as error:
            timings['execute_total']=time.monotonic()-started
            self.robot.audit('primitive_execute_failed',plan_id=plan_id,status='failed',
                             error=str(error),timings_s=dict(timings))
            raise
        finally:
            if self.moveit is not None:
                self.moveit.execution_reference = None
        timings['execute_total']=time.monotonic()-started
        result['timings_s']=timings
        if result['status']=='completed':self.robot.audit('primitive_completed',**result)
        else:
            self.robot.audit('primitive_failed',error=result['message'],**result)
            if result['recoverable']:self.robot.audit('primitive_recovery_available',**result)
        return result

    def _execute(self,plan_id,timings,started):
        candidate=self.prepared
        try:
            with _measure(timings,'execution_validation'):
                prepared=self._validate_execution(plan_id)
        except PeerJointMovedError as error:
            return undispatched_peer_failure(plan_id,candidate,error,self.robot.motion_remaining,
                                             timings['execution_validation'])
        token,created,initial,stages,paths,operations,skill,holding=prepared;r=self.robot;p=self.profile
        r.motion_remaining-=operations;receipts=[];grasp_verified=False
        r.audit('primitive_dispatch',plan_id=token,skill=skill,stages=stages,timings_s=dict(timings))
        stage=None;stage_started=None;stage_timings={};departed=False;actuation_attempted=False
        try:
            for stage_index,(stage,path) in enumerate(zip(stages,paths)):
                stage_started=time.monotonic();stage_timings={};command_state='not_dispatched'
                phase=stage_phase(stages,stage_index,skill);motion_detail=None;moveit_receipt=None
                r.audit('primitive_stage_started',plan_id=token,stage_index=stage_index,
                        skill=skill,backend='primitives',phase=phase,stage=stage)
                with _measure(stage_timings,'stage_validation',timings):
                    r.ready();initial=r.joints()
                if self.moveit is not None:
                    with _measure(stage_timings,'moveit_stage',timings):
                        moveit_receipt = self.moveit.check(initial, [(stage_index, phase, path)], executing=True)
                        measured = r.joints()
                        if max(abs(measured[n]-initial[n]) for n in initial)>.015:
                            raise RuntimeError('Arm moved during MoveIt trajectory revalidation')
                if stage['kind']=='move':
                    with _measure(stage_timings,'motion_verification',timings):
                        if holding and not r.grip.gripping():raise RuntimeError('Held object lost before movement')
                    options={}
                    continuous=r.cfg.get('planning',{}).get('trajectory_mode')=='continuous'
                    if continuous:
                        with _measure(stage_timings,'trajectory_preflight',timings):
                            departing=skill=='pick' and grasp_verified and not departed
                            self._check_curve(initial,path,holding,departing,phase)
                            peer_check=_peer_path_check(r,skill,holding)
                            if callable(peer_check):peer_check(r,[path])
                            # Preflight may be expensive; do not start from a pose which changed meanwhile.
                            if max(abs(r.joints()[n]-initial[n]) for n in initial)>.015:
                                raise RuntimeError('Arm moved during trajectory revalidation')
                            if departing:departed=True
                        options={'continuous':True,'motion_phase':phase}
                    r.motion_context={'plan_id':token,'stage_index':stage_index,'skill':skill,'backend':'primitives','phase':phase}
                    with _measure(stage_timings,'motion',timings):
                        start_position=r.endpoint_position()
                        actuation_attempted=True
                        command_state='outcome_unknown'
                        motion_detail=r.move_segment(initial,path,start_position,max_travel=p['max_leg_m'],**options)
                        command_state='command_returned'
                    with _measure(stage_timings,'motion_verification',timings):
                        d,a=r.arrival_error(path[-1][1])
                        if d>r.cfg.get('arrival_position_tolerance_m',.01) or a>.04:
                            raise RuntimeError('Primitive stage did not reach target; do not proceed to contact/release')
                        if holding and not r.grip.gripping():raise RuntimeError('Held object lost during movement')
                elif stage['kind']=='gripper':
                    with _measure(stage_timings,'gripper_command',timings):
                        actuation_attempted=True
                        command_state='outcome_unknown'
                        if not r.grip.command_position(100. if stage['open'] else 0.,block=True,timeout=2):
                            raise RuntimeError('Gripper command timeout')
                        command_state='command_returned'
                    with _measure(stage_timings,'gripper_settle',timings):
                        time.sleep(.25)
                    with _measure(stage_timings,'gripper_verification',timings):
                        if stage['open'] and r.grip.position()<70:raise RuntimeError('Gripper failed to open')
                        holding=bool(r.grip.gripping())
                else:
                    with _measure(stage_timings,'grasp_verification',timings):
                        if bool(r.grip.gripping()) != stage['held']:
                            if skill=='pick' and stage['held'] and not grasp_verified and receipts and receipts[-1]['stage']=={'kind':'gripper','open':False}:
                                raise EmptyGrasp('Empty grasp confirmed after close; retract and reobserve before retry')
                            raise RuntimeError('Held object lost or release not confirmed')
                        if stage['held']:grasp_verified=True
                with _measure(stage_timings,'stage_validation',timings):
                    r.ready()
                    receipt={'stage':stage,'eef_m':r.endpoint_position(),'gripping':bool(r.grip.gripping())}
                receipt.update(stage_index=stage_index,phase=phase,duration_s=time.monotonic()-stage_started,timings_s=dict(stage_timings))
                if isinstance(motion_detail,dict):receipt['motion_detail']=motion_detail
                if moveit_receipt is not None:receipt['moveit']=moveit_receipt
                receipts.append(receipt)
                stage_started=None
                r.audit('primitive_stage_applied',plan_id=token,**receipt)
        except BaseException as error:
            failed_stage=None
            if stage_started is not None:
                failed_stage={'stage':stage,'stage_index':stage_index,'phase':phase,'status':'failed','duration_s':time.monotonic()-stage_started,
                              'timings_s':dict(stage_timings),'command_state':command_state}
            stop_attempted=self.moveit is None or actuation_attempted
            stopped=None;stop_error=None
            if not stop_attempted:
                r.motion_remaining+=operations
            else:
                stopped=False
                with _measure(timings,'stop'):
                    stop_errors=[]
                    try:r.limb.set_joint_positions(r.joints())
                    except Exception as failure:stop_errors.append('arm hold: '+str(failure))
                    try:
                        if r.grip.stop(block=False) is False:raise RuntimeError('Gripper stop rejected')
                    except Exception as failure:stop_errors.append('gripper stop: '+str(failure))
                    try:r.ready()
                    except Exception as failure:stop_errors.append('stop verification: '+str(failure))
                    stopped=not stop_errors
                    if stop_errors:
                        stop_error='; '.join(stop_errors);r.ros.logwarn('Primitive stop failed: %s',stop_error)
            if not isinstance(error,Exception):raise
            measured_commands=sum(item['stage']['kind'] in ('move','gripper') for item in receipts)
            uncertain=bool(failed_stage and failed_stage['command_state']=='outcome_unknown')
            applied=(True if measured_commands or (failed_stage and command_state=='command_returned')
                     else None if uncertain else False)
            empty=isinstance(error,EmptyGrasp)
            return {'plan_id':token,'skill':skill,'status':'failed','failure_code':'empty_grasp' if empty else 'execution_failed',
                    'failure_type':type(error).__name__,'message':str(error),'applied':applied,
                    'completed_stages':receipts,'completed_stage_count':len(receipts),'confirmed_commands':measured_commands,
                    'failed_stage':failed_stage,'outcome_uncertain':uncertain or stopped is False,'recoverable':bool(empty and stopped),
                    'retryable':False,'stop_attempted':stop_attempted,'stopped':stopped,'stop_error':stop_error,
                    'gripping':False if empty else None,'motion_remaining':r.motion_remaining}
        finally:
            if actuation_attempted:
                r.wrist_after_motion=time.monotonic()
        return {'plan_id':token,'skill':skill,'applied':True,'status':'completed','completed_stages':receipts,
                'gripping':bool(r.grip.gripping()),'motion_remaining':r.motion_remaining}

    def _validate_execution(self,plan_id):
        prepared,self.prepared=self.prepared,None
        if prepared is None or prepared[0]!=plan_id:raise ValueError('Plan consumed or superseded; never replay')
        token,created,initial,stages,paths,operations,skill,holding=prepared;r=self.robot;p=self.profile
        if not r.armed or r.cfg.get('commissioning_scope') not in ('supervised_primitive_task','supervised_hybrid_task'):raise RuntimeError('Primitive bridge is read-only or wrong scope')
        if p.get('calibrated') is not True:raise RuntimeError('Primitive geometry is not calibrated')
        if time.monotonic()-created>r.cfg['max_observation_age_s']:raise RuntimeError('Plan expired')
        if max(abs(r.joints()[k]-initial[k]) for k in initial)>.015:raise RuntimeError('Arm moved since planning')
        if r.motion_remaining<operations:raise RuntimeError('Stage budget exhausted')
        r.ready()
        return prepared
