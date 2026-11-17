"""One camera and one serialized RPC owner for alternating Baxter arm primitives.

Only one arm executes at a time. Table hand-offs use place, observation and pick;
coordinated two-arm lifting and simultaneous motor streams are not implemented.
"""
import argparse
import copy
import hashlib
import json
import math
import threading
import time
from contextlib import ExitStack
from pathlib import Path

from .baxter import Baxter, validate_config
from .camera import D455Camera
from .collision import (ArmCollisionError, ClearanceUnavailableError, InterArmGuard,
                        KDLArmCapsules, PeerStateChangedError, validate_clearance_config)
from .primitive_service import PrimitiveService
from .primitives import SKILLS, tcp_from_eef
from .rpc import serve
from .workspace import workspace_metadata

ARMS=('left','right')
PROTOCOL='serialized_dual_primitives_v1'


def build_dual_candidate(left,right,budget=64,lease_seconds=600,clearance=None):
    """Assemble an explicitly unarmed candidate; never promote geometric evidence."""
    profiles={arm:copy.deepcopy(profile) for arm,profile in zip(ARMS,(left,right))}
    serial=profiles['left'].get('camera_serial','')
    for arm,profile in profiles.items():
        profile.update(arm=arm,commissioned=False,commissioning_scope='supervised_primitive_task',
                       motion_budget=budget,lease_seconds=lease_seconds)
        profile['primitives']['calibrated']=False
    candidate={'camera_serial':serial,'motion_budget':budget,'lease_seconds':lease_seconds,
               'arms':profiles,'inter_arm_clearance':copy.deepcopy(clearance)}
    if candidate['inter_arm_clearance'] is not None:candidate['inter_arm_clearance']['verified']=False
    return validate_dual_config(candidate,False)


def validate_dual_config(cfg, armed=False):
    if not isinstance(cfg,dict) or set(cfg.get('arms',{}))!=set(ARMS):
        raise ValueError('Dual configuration requires complete left and right arm profiles')
    if not isinstance(cfg.get('camera_serial',''),str):raise ValueError('Invalid shared camera serial')
    if type(cfg.get('motion_budget')) is not int or not 1<=cfg['motion_budget']<=64:
        raise ValueError('Dual task requires one shared stage budget in 1..64')
    lease=cfg.get('lease_seconds')
    if type(lease) not in (int,float) or not math.isfinite(lease) or not 1<=lease<=600:
        raise ValueError('Dual task requires one shared lease of at most 600 seconds')
    for arm in ARMS:
        profile=cfg['arms'][arm]
        if profile.get('arm')!=arm or profile.get('commissioning_scope')!='supervised_primitive_task':
            raise ValueError('Each dual arm uses its own supervised_primitive_task profile')
        if profile.get('motion_budget')!=cfg['motion_budget'] or profile.get('lease_seconds')!=lease:
            raise ValueError('Per-arm budgets/leases must match the shared task, not add to it')
        if profile.get('camera_serial','')!=cfg.get('camera_serial',''):
            raise ValueError('Both arm profiles must use the single shared camera')
        validate_config(profile,armed)
    # One physical camera has one base transform. Tool/TCP and workspaces may differ.
    for key in ('camera_to_base_xyzw','camera_to_base_translation_m'):
        if cfg['arms']['left']['primitives'][key]!=cfg['arms']['right']['primitives'][key]:
            raise ValueError('Both arms must share the same camera-to-base calibration')
    validate_clearance_config(cfg.get('inter_arm_clearance'),armed)
    return cfg


class SharedMotion:
    def __init__(self,budget,lease_seconds):
        self.remaining=budget
        self.deadline=time.monotonic()+lease_seconds


class DualArmBaxter(Baxter):
    def __init__(self,cfg,armed,*,camera,shared):
        self.shared=shared
        self.dual_owner=None
        super().__init__(cfg,armed,camera=camera)

    @property
    def motion_remaining(self):return self.shared.remaining

    @motion_remaining.setter
    def motion_remaining(self,value):
        if type(value) is not int or value<0:raise ValueError('Invalid shared stage budget')
        self.shared.remaining=value

    @property
    def lease_end(self):return self.shared.deadline

    @lease_end.setter
    def lease_end(self,value):self.shared.deadline=min(self.shared.deadline,value)

    def ready(self):
        super().ready()
        if self.dual_owner is not None:self.dual_owner.check_running_arm(self.cfg['arm'])

    def check_other_arm_path(self,paths):
        if self.dual_owner is None:raise ClearanceUnavailableError('No dual execution owner')
        return self.dual_owner.check_other_arm_path(self.cfg['arm'],paths)


class DualBridge:
    def __init__(self,robots,cfg,shared,guard=None,service_factory=PrimitiveService):
        validate_dual_config(cfg,any(r.armed for r in robots.values()))
        if set(robots)!=set(ARMS):raise ValueError('Both robot interfaces are required')
        if robots['left'].camera is not robots['right'].camera:
            raise ValueError('Dual arm interfaces must share one camera owner')
        if robots['left'].armed!=robots['right'].armed:
            raise ValueError('Mixed armed/read-only interfaces are not a dual bridge')
        self.robots=robots;self.cfg=cfg;self.shared=shared;self.guard=guard
        self.armed=robots['left'].armed
        if self.armed and guard is None:raise ClearanceUnavailableError('Armed dual bridge has no collision guard')
        self.services={arm:service_factory(robots[arm]) for arm in ARMS}
        self.selected_arm=None;self.peer_reference=None;self.clearance_receipt=None;self.clearance_seconds=0.
        self.executing_arm=None;self.lock=threading.RLock();self.closed=False
        for robot in robots.values():robot.dual_owner=self

    def _arm(self,arm):
        if arm not in ARMS:raise ValueError('Select robot arm left or right explicitly')
        return arm

    def _invalidate(self,keep_observation=None,keep_prepared=None):
        for arm,service in self.services.items():
            self.robots[arm].pending=None
            if arm!=keep_observation:service.observation=None
            if arm!=keep_prepared:service.prepared=None

    def _describe_arm(self,arm,state):
        import pyrealsense2 as rs
        robot=self.robots[arm];profile=robot.cfg['primitives']
        tcp=tcp_from_eef(state['eef_m'],state['eef_xyzw'],profile)
        recovery={'view_indices':[i for i,v in enumerate(profile.get('observation_tcp_m',[])) if math.dist(tcp,v)>=.02],
                  'motion_remaining':self.shared.remaining}
        skills=[s for s in SKILLS if (s!='clear_view' or recovery['view_indices'])
                and (s!='home' or profile.get('home_tcp_m') is not None)]
        workspace=workspace_metadata(profile,robot.camera.intrinsics,rs.rs2_project_point_to_pixel)
        return workspace,skills,recovery

    def health(self):
        with self.lock:
            per_arm={arm:self.robots[arm].health() for arm in ARMS}
            geometry={'arms':{arm:v.get('environment_fingerprint') for arm,v in per_arm.items()},
                      'inter_arm_clearance':self.cfg.get('inter_arm_clearance')}
            fingerprint=hashlib.sha256(json.dumps(geometry,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:24]
            return {'armed':self.armed,'motion_scope':'supervised_dual_primitive_task','dual_protocol':PROTOCOL,'protocol':PROTOCOL,
                    'backend_capabilities':['primitives'],'available_arms':list(ARMS),'selected_arm':self.selected_arm,
                    'motion_remaining':self.shared.remaining,'lease_remaining_s':max(0.,self.shared.deadline-time.monotonic()),
                    'environment_fingerprint':fingerprint,'arms':per_arm,
                    'clearance':{'configured':self.guard is not None,
                                 'verified':bool((self.cfg.get('inter_arm_clearance') or {}).get('verified')),
                                 'model':(self.cfg.get('inter_arm_clearance') or {}).get('body_geometry','urdf_chain_capsules'),
                                 'links':({arm:list(self.guard.models[arm].links) for arm in ARMS} if self.guard else {})}}

    def observe(self,arm):
        began=time.monotonic()
        with self.lock:
            self._invalidate();self.peer_reference=None;self.clearance_receipt=None
            self.selected_arm=self._arm(arm)
            try:
                obs=self.services[arm].observe()
                peer='right' if arm=='left' else 'left'
                before_peer=time.monotonic()
                states={arm:obs['state'],peer:self.robots[peer].read_state()}
                # The chosen arm must still have fresh endpoint/joint telemetry after the other read.
                self.robots[arm].joints()
                after_peer=time.monotonic()
                self.peer_reference=copy.deepcopy(states[peer])
                workspaces={};skills={};recovery={}
                for selected in ARMS:
                    workspaces[selected],skills[selected],recovery[selected]=self._describe_arm(selected,states[selected])
                obs.update(selected_arm=arm,available_arms=list(ARMS),arm_states=states,
                           workspaces=workspaces,arm_skills=skills,arm_recovery=recovery)
                obs.setdefault('timings_s',{}).update(dual_peer_state_read=after_peer-before_peer,
                                                     dual_arm_metadata=time.monotonic()-after_peer,
                                                     dual_observe_total=time.monotonic()-began)
                return obs
            except BaseException:
                self._invalidate();self.peer_reference=None
                raise

    def prepare(self,arm,obs_id,skill,uv=None,target_base_m=None,view_index=0):
        began=time.monotonic()
        with self.lock:
            keep=arm if arm==self.selected_arm and arm in ARMS else None
            self._invalidate(keep_observation=keep);self.clearance_receipt=None;self.clearance_seconds=0.
            self._arm(arm)
            if keep is None:
                self.peer_reference=None
                raise ValueError('Observe the selected arm before planning')
            try:
                result=self.services[arm].prepare(obs_id,skill,uv,target_base_m,view_index)
            except (ArmCollisionError,PeerStateChangedError,ClearanceUnavailableError) as error:
                self._invalidate()
                code=('inter_arm_clearance' if isinstance(error,ArmCollisionError) else
                      'peer_changed' if isinstance(error,PeerStateChangedError) else 'clearance_unavailable')
                result={'status':'rejected','obs_id':obs_id,'skill':skill,'applied':False,
                        'failure_code':code,'message':str(error),'recoverable':code!='clearance_unavailable'}
                self.robots[arm].audit('dual_prepare_rejected',selected_arm=arm,**result)
            except BaseException:
                self._invalidate();raise
            if result.get('status')=='ready':
                if self.clearance_receipt is None:
                    self._invalidate()
                    raise ClearanceUnavailableError('Primitive planner did not run the full-path inter-arm hook')
                result['inter_arm_clearance']=dict(self.clearance_receipt)
            result['selected_arm']=arm
            result.setdefault('timings_s',{}).update(dual_prepare_total=time.monotonic()-began,
                                                    dual_collision_preflight=self.clearance_seconds)
            return result

    def check_other_arm_path(self,arm,paths):
        if arm!=self.selected_arm or self.peer_reference is None:
            raise PeerStateChangedError('Missing inactive-arm observation')
        if self.guard is None:raise ClearanceUnavailableError('Configure full-link radii and installed tool capsules before planning')
        peer='right' if arm=='left' else 'left'
        began=time.monotonic()
        try:
            self.clearance_receipt=self.guard.path_clearance(arm,self.robots[arm].joints(),paths,
                                                            self.peer_reference,self.robots[peer].read_state())
        finally:self.clearance_seconds=time.monotonic()-began
        return self.clearance_receipt

    def check_running_arm(self,arm):
        if self.executing_arm!=arm or self.peer_reference is None or self.clearance_receipt is None:
            raise PeerStateChangedError('No current serialized dual execution lease')
        if time.monotonic()>=self.shared.deadline:raise RuntimeError('Shared dual motion lease expired')
        if self.guard is None:raise ClearanceUnavailableError('Missing inter-arm collision guard')
        peer='right' if arm=='left' else 'left'
        peer_state=self.robots[peer].read_state()
        self.guard.check_peer(self.peer_reference,peer_state)
        self.guard.current_clearance(arm,self.robots[arm].read_state(),peer_state)

    def execute(self,arm,plan_id):
        began=time.monotonic()
        with self.lock:
            keep=arm if arm in ARMS and arm==self.selected_arm else None
            self._invalidate(keep_prepared=keep)
            try:
                self._arm(arm)
                if not self.armed:raise RuntimeError('Dual bridge is read-only')
                if keep is None:raise ValueError('Selected arm changed; obtain a new observation and plan')
                self.executing_arm=arm
                self.check_running_arm(arm)
                result=self.services[arm].execute(plan_id)
                result['selected_arm']=arm
                result['arm_states']={selected:self.robots[selected].read_state() for selected in ARMS}
                result.setdefault('timings_s',{})['dual_execute_total']=time.monotonic()-began
                return result
            finally:
                self.executing_arm=None;self.peer_reference=None;self.clearance_receipt=None
                self._invalidate()

    def methods(self):
        return {'healthz':self.health,'dual.observe':self.observe,'dual.prepare':self.prepare,'dual.execute':self.execute,
                'dual.scene_snapshot':self.scene_snapshot}

    def scene_snapshot(self,arm):
        """Capture fresh aligned RGB-D for onsite geometry checks using the shared camera."""
        with self.lock:
            obs=self.observe(arm)
            folder=Path('runs/geometry');folder.mkdir(parents=True,exist_ok=True)
            path=folder/(obs['id']+'.npz')
            self.robots[arm].camera.save_depth(str(path))
            return {'obs_id':obs['id'],'selected_arm':arm,'state':obs['state'],
                    'arm_states':obs['arm_states'],'depth_file':str(path.resolve())}

    def close(self):
        # Close subscribers before releasing the shared USB stream exactly once.
        if self.closed:return
        self.closed=True
        with ExitStack() as stack:
            stack.callback(self.robots['left'].camera.close)
            for robot in self.robots.values():stack.callback(robot.close)


def create_bridge(cfg,armed=False,*,camera_factory=D455Camera,robot_factory=DualArmBaxter,model_factory=KDLArmCapsules):
    validate_dual_config(cfg,armed)
    shared=SharedMotion(cfg['motion_budget'],cfg['lease_seconds'])
    camera=camera_factory(cfg.get('camera_serial',''));robots={}
    with ExitStack() as stack:
        stack.callback(camera.close)
        for arm in ARMS:
            robots[arm]=robot_factory(cfg['arms'][arm],armed,camera=camera,shared=shared)
            stack.callback(robots[arm].close)
        clearance=cfg.get('inter_arm_clearance')
        guard=InterArmGuard(clearance,{arm:model_factory(arm,clearance) for arm in ARMS}) if clearance is not None else None
        bridge=DualBridge(robots,cfg,shared,guard)
        stack.pop_all()
        return bridge


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--arm-motion',action='store_true')
    parser.add_argument('--port',type=int,default=9102)
    args=parser.parse_args()
    cfg=json.loads(Path(args.config).read_text(encoding='utf-8'))
    validate_dual_config(cfg,args.arm_motion)
    import rospy
    import signal
    rospy.init_node('cix_baxter_dual_bridge',disable_signals=True)
    def terminate(signum,frame):raise SystemExit(0)
    signal.signal(signal.SIGTERM,terminate)
    bridge=create_bridge(cfg,args.arm_motion)
    try:serve(args.port,bridge.methods())
    finally:
        try:bridge.close()
        finally:rospy.signal_shutdown('dual bridge stopped')


if __name__=='__main__':main()
