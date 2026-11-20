"""Offline ownership, freshness and full-link clearance checks for serial dual arms."""
import copy
import json
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch

from cix_harness.collision import (ArmCollisionError,Capsule,ClearanceUnavailableError,
                                  InterArmGuard,KDLArmCapsules,PeerStateChangedError,check_capsules,
                                  segment_distance,validate_clearance_config,arm_chain_links,build_collision_candidate)
from cix_harness.baxter import Baxter
from cix_harness.dual_bridge import (DualArmBaxter,DualBridge,SharedMotion,
                                    create_bridge,validate_dual_config,build_dual_candidate)


def configuration(armed=False):
    base=json.loads(Path('config/baxter-primitives.example.json').read_text())
    base.update(commissioned=armed,workspace_min_m=[.3,-.8,-.16],workspace_max_m=[1.1,.8,.5],motion_budget=12,lease_seconds=600)
    base['primitives']['calibrated']=armed
    base['primitives']['tcp_workspace_min_m'][2]=-.17
    profiles={arm:{**copy.deepcopy(base),'arm':arm} for arm in ('left','right')}
    return {'camera_serial':'','motion_budget':12,'lease_seconds':600,'arms':profiles,
            'inter_arm_clearance':{'verified':armed,'minimum_distance_m':.04,
                'max_peer_joint_drift_rad':.001,'max_joint_sample_rad':.02,
                'link_radii_m':{arm:{arm+'_upper_shoulder':.03,arm+'_wrist':.03} for arm in profiles},
                'tool_capsules':{arm:[{'start_m':[0,0,0],'end_m':[0,0,.1],'radius_m':.03}] for arm in profiles},
                'held_object_radius_m':{arm:.04 for arm in profiles}}}


class FakeCamera:
    def __init__(self,serial=''):
        self.captures=0;self.closes=0;self.intrinsics=NS()
        self.last_depth=NS(copy=lambda:'depth')
    def close(self):self.closes+=1


class FakeArm(DualArmBaxter):
    def __init__(self,cfg,armed,*,camera,shared):
        self.cfg=cfg;self.armed=armed;self.camera=camera;self.shared=shared;self.dual_owner=None
        self.pending=None;self.stale=False;self.closed=False;self.moves=[];self.owns_camera=False
        self.names=[cfg['arm']+'_j'+str(i) for i in range(7)]
        self.angles={name:0. for name in self.names}
        self.position=[.6,.2 if cfg['arm']=='left' else -.2,.1]
        self.grip=Mock();self.grip.gripping.return_value=False
        self.grip.position.return_value=100.;self.grip.command_position.return_value=True;self.grip.stop.return_value=True
        self.limb=Mock();self.limb.endpoint_pose.side_effect=self.endpoint_pose
        self.ros=Mock();self.audit=Mock()
        self.continuous_ik=Mock();self.continuous_ik.solve.side_effect=lambda target,seed,observed:dict(seed)
        self.arrival_error=Mock(return_value=(0.,0.))
    def close(self):self.closed=True
    def joints(self):
        if self.stale:raise RuntimeError('Stale arm telemetry')
        return dict(self.angles)
    def endpoint_pose(self):
        return {'position':NS(x=self.position[0],y=self.position[1],z=self.position[2]),
                'orientation':NS(x=1.,y=0.,z=0.,w=0.)}
    def endpoint_position(self):return list(self.position)
    def read_state(self,frame_number=None):
        angles=self.joints()
        return {'arm':self.cfg['arm'],'eef_m':list(self.position),'eef_xyzw':[1,0,0,0],
                'joint_names':list(self.names),'joint_rad':[angles[n] for n in self.names],
                'gripping':self.grip.gripping(),'gripper_percent':self.grip.position()}
    def observe(self):
        self.camera.captures+=1
        return {'id':self.cfg['arm']+'-'+str(self.camera.captures),'image':'offline-image','state':self.read_state()}
    def health(self):return {'armed':self.armed,'motion_remaining':self.motion_remaining,'environment_fingerprint':self.cfg['arm']}
    def ready(self):
        if time.monotonic()>=self.lease_end:raise RuntimeError('Shared lease expired')
        if self.dual_owner:self.dual_owner.check_running_arm(self.cfg['arm'])
    def inside(self,xyz):pass
    def guard_pose(self,pose):pass
    def move_segment(self,initial,path,start,max_travel):
        self.ready();self.moves.append(path)
        if path:
            self.angles.update(path[-1][0]);p=path[-1][1].pose.position
            self.position=[p.x,p.y,p.z]


class FakeModel:
    def __init__(self,arm,cfg):self.arm=arm;self.links=list(cfg['link_radii_m'][arm]);self.reach_bound=.2
    def capsules(self,joints,holding=False):
        y=.4 if self.arm=='left' else -.4
        return [Capsule(self.arm+'_forearm',(0,y,0),(1,y,0),.03),
                Capsule(self.arm+'_tool',(1,y,0),(1,y,.1),.07 if holding else .03)]


class DualBridgeTests(unittest.TestCase):
    def setUp(self):
        geometry=NS(PoseStamped=lambda:NS(header=NS(),pose=NS(position=NS(),orientation=NS())))
        modules={'geometry_msgs.msg':geometry,'tf':NS(transformations=NS(quaternion_slerp=lambda a,b,f:a)),
                 'pyrealsense2':NS(rs2_project_point_to_pixel=Mock())}
        self.patches=[patch.dict('sys.modules',modules),patch('cix_harness.primitive_service.time.sleep'),
                      patch('cix_harness.primitive_service.workspace_metadata',return_value={'frame':'base'}),
                      patch('cix_harness.dual_bridge.workspace_metadata',return_value={'frame':'base'})]
        for item in self.patches:item.start();self.addCleanup(item.stop)
    def bridge(self,armed=False,config=None):
        bridge=create_bridge(config or configuration(armed),armed,camera_factory=FakeCamera,robot_factory=FakeArm,model_factory=FakeModel)
        self.addCleanup(bridge.close)
        return bridge
    def plan(self,bridge,arm,skill='open',target=None):
        obs=bridge.observe(arm)
        return bridge.prepare(arm,obs['id'],skill,target_base_m=target)

    def test_one_camera_selected_state_both_fresh_arm_contexts_and_idempotent_close(self):
        bridge=self.bridge();camera=bridge.robots['left'].camera
        obs=bridge.observe('right')
        self.assertIs(camera,bridge.robots['right'].camera);self.assertEqual(camera.captures,1)
        self.assertEqual(obs['state']['arm'],'right');self.assertEqual(obs['selected_arm'],'right')
        self.assertEqual(set(obs['arm_states']),{'left','right'})
        for field in ('workspaces','arm_skills','arm_recovery'):self.assertEqual(set(obs[field]),{'left','right'})
        self.assertEqual(bridge.health()['motion_scope'],'supervised_dual_primitive_task')
        self.assertEqual(bridge.health()['backend_capabilities'],['primitives'])
        self.assertEqual(set(bridge.methods()),{'healthz','dual.observe','dual.prepare','dual.execute','dual.scene_snapshot'})
        bridge.close();bridge.close();self.assertEqual(camera.closes,1)

    def test_home_is_not_advertised_when_profile_has_no_site_home(self):
        cfg=configuration();cfg['arms']['right']['primitives']['home_tcp_m']=None
        bridge=self.bridge(config=cfg);obs=bridge.observe('right')
        self.assertNotIn('home',obs['primitive_skills']);self.assertNotIn('home',obs['arm_skills']['right'])
        self.assertIn('home',obs['arm_skills']['left'])

    def test_left_only_scene_keeps_the_tagged_right_arm_passive(self):
        cfg=configuration(True)
        cfg['motion_arms']=['left']
        cfg['arms']['right']['commissioned']=False
        cfg['arms']['right']['primitives']['calibrated']=False
        bridge=self.bridge(True,config=cfg)
        self.assertTrue(bridge.robots['left'].armed)
        self.assertFalse(bridge.robots['right'].armed)
        self.assertEqual(bridge.health()['available_arms'],['left'])
        self.assertIn('right',bridge.observe('left')['arm_states'])
        for call in (lambda:bridge.observe('right'),lambda:bridge.prepare('right','x','open'),
                     lambda:bridge.execute('right','x')):
            with self.assertRaisesRegex(ValueError,'passive'):call()
        plan=self.plan(bridge,'left')
        self.assertEqual(bridge.execute('left',plan['plan_id'])['status'],'completed')
        bridge.robots['right'].grip.command_position.assert_not_called()

    def test_depth_snapshot_uses_shared_camera_and_revokes_previous_plan(self):
        import tempfile
        bridge=self.bridge();self.plan(bridge,'left')
        camera=bridge.robots['left'].camera;camera.save_depth=Mock()
        with tempfile.TemporaryDirectory() as folder,patch('cix_harness.dual_bridge.Path',return_value=Path(folder)):
            result=bridge.scene_snapshot('right')
        self.assertEqual(result['selected_arm'],'right')
        self.assertEqual(camera.captures,2)
        camera.save_depth.assert_called_once()
        self.assertTrue(result['depth_file'].endswith('right-2.npz'))
        for arm,service in bridge.services.items():
            self.assertIsNone(service.prepared)
            bridge.robots[arm].grip.command_position.assert_not_called()

    def test_camera_only_snapshot_does_not_require_or_certify_gripper_telemetry(self):
        import tempfile
        bridge=self.bridge();self.plan(bridge,'left')
        arm=bridge.robots['left'];arm.stale=True
        arm.camera.capture=Mock(return_value=(None,0.,42));arm.camera.save_depth=Mock()
        with tempfile.TemporaryDirectory() as folder,patch('cix_harness.dual_bridge.Path',return_value=Path(folder)):
            snapshot=bridge.scene_snapshot('left',camera_only=True)
        self.assertEqual(snapshot['frame_number'],42)
        self.assertNotIn('state',snapshot)
        self.assertIsNone(bridge.services['left'].prepared)
        self.assertIsNone(bridge.services['left'].observation)
        arm.camera.save_depth.assert_called_once()

    def test_dual_constructor_initial_ready_has_no_owner_but_later_ready_checks_peer(self):
        cfg=configuration(True)['arms']['left'];shared=SharedMotion(12,600)
        def initialize(robot,cfg,armed,*,camera):
            robot.cfg=cfg;robot.camera=camera;robot.armed=armed
            robot.ready()
        with patch.object(Baxter,'__init__',initialize),patch.object(Baxter,'ready') as hardware_ready:
            robot=DualArmBaxter(cfg,True,camera=FakeCamera(),shared=shared)
            self.assertIsNone(robot.dual_owner);hardware_ready.assert_called_once()
            owner=Mock();robot.dual_owner=owner;robot.ready()
            owner.check_running_arm.assert_called_once_with('left')

    def test_factory_failure_releases_the_one_camera_and_prior_arm_subscribers(self):
        camera=FakeCamera();created=[]
        def factory(cfg,armed,**kwargs):
            if cfg['arm']=='right':raise RuntimeError('right telemetry unavailable')
            arm=FakeArm(cfg,armed,**kwargs);created.append(arm);return arm
        with self.assertRaisesRegex(RuntimeError,'right telemetry'):
            create_bridge(configuration(),camera_factory=lambda _:camera,robot_factory=factory,model_factory=FakeModel)
        self.assertTrue(created[0].closed);self.assertEqual(camera.closes,1)

    def test_stale_inactive_arm_invalidates_observation_and_all_plans(self):
        bridge=self.bridge();self.plan(bridge,'left')
        bridge.robots['right'].stale=True
        with self.assertRaisesRegex(RuntimeError,'Stale'):bridge.observe('left')
        for service in bridge.services.values():
            self.assertIsNone(service.observation);self.assertIsNone(service.prepared)

    def test_budget_is_shared_when_alternating_arms_and_lease_cannot_be_reset(self):
        bridge=self.bridge(True);deadline=bridge.shared.deadline
        for arm in ('left','right'):
            prepared=self.plan(bridge,arm)
            self.assertTrue(prepared['inter_arm_clearance']['checked'])
            result=bridge.execute(arm,prepared['plan_id'])
            self.assertEqual(result['status'],'completed')
        self.assertEqual(bridge.shared.remaining,10)
        self.assertEqual([r.motion_remaining for r in bridge.robots.values()],[10,10])
        bridge.robots['right'].lease_end=time.monotonic()+1000
        self.assertEqual(bridge.shared.deadline,deadline)
        bridge.shared.deadline=time.monotonic()-1
        prepared=self.plan(bridge,'left')
        with self.assertRaisesRegex(RuntimeError,'lease'):bridge.execute('left',prepared['plan_id'])
        self.assertEqual(bridge.shared.remaining,10)

    def test_observing_other_arm_revokes_old_token_and_duplicate_execution(self):
        bridge=self.bridge(True);old=self.plan(bridge,'left')
        bridge.observe('right')
        with self.assertRaisesRegex(ValueError,'changed'):bridge.execute('left',old['plan_id'])
        bridge.robots['left'].grip.command_position.assert_not_called()
        current=self.plan(bridge,'right');bridge.execute('right',current['plan_id'])
        with self.assertRaises(RuntimeError):bridge.execute('right',current['plan_id'])
        self.assertEqual(bridge.robots['right'].grip.command_position.call_count,1)

    def test_invalid_arm_execution_also_consumes_prepared_tokens(self):
        bridge=self.bridge(True);self.plan(bridge,'left')
        with self.assertRaisesRegex(ValueError,'Select robot arm'):bridge.execute('both','wrong')
        self.assertIsNone(bridge.services['left'].prepared)

    def test_readonly_missing_clearance_observes_but_cannot_issue_ready_plan(self):
        cfg=configuration();cfg.pop('inter_arm_clearance');bridge=self.bridge(config=cfg)
        obs=bridge.observe('left');result=bridge.prepare('left',obs['id'],'open')
        self.assertEqual(result['failure_code'],'clearance_unavailable');self.assertFalse(result['applied'])
        self.assertFalse(result['recoverable']);self.assertIsNone(bridge.services['left'].prepared)
        with self.assertRaisesRegex(RuntimeError,'read-only'):bridge.execute('left','anything')
        self.assertFalse(bridge.health()['clearance']['configured'])

    def test_armed_missing_or_unverified_clearance_fails_before_camera_is_opened(self):
        for cfg in (configuration(True),configuration(True)):
            if cfg['inter_arm_clearance']['verified']:cfg['inter_arm_clearance']['verified']=False
            camera=Mock()
            with self.assertRaisesRegex(ValueError,'verified'):
                create_bridge(cfg,True,camera_factory=camera,robot_factory=FakeArm,model_factory=FakeModel)
            camera.assert_not_called()
        cfg=configuration(True);cfg.pop('inter_arm_clearance')
        with self.assertRaisesRegex(ValueError,'requires verified'):validate_dual_config(cfg,True)

    def test_peer_joint_or_gripper_change_after_prepare_prevents_any_actuation(self):
        for change in ('joint','gripper'):
            bridge=self.bridge(True);prepared=self.plan(bridge,'left');right=bridge.robots['right']
            if change=='joint':right.angles[right.names[0]]+=.02
            else:right.grip.gripping.return_value=True
            if change=='joint':
                result=bridge.execute('left',prepared['plan_id'])
                self.assertFalse(result['applied'])
                self.assertEqual(result['confirmed_commands'],0)
                self.assertEqual(result['failure_type'],'PeerJointMovedError')
                self.assertEqual(result['failed_stage']['command_state'],'not_dispatched')
                self.assertEqual(result['failed_stage']['phase'],'dispatch_validation')
                self.assertIsNone(bridge.services['left'].prepared)
            else:
                with self.assertRaises(PeerStateChangedError):bridge.execute('left',prepared['plan_id'])
            for robot in bridge.robots.values():robot.grip.command_position.assert_not_called()
            self.assertEqual(bridge.shared.remaining,12)

    def test_running_reference_advances_only_after_the_new_segment_and_peer_are_checked(self):
        bridge=self.bridge(True)
        self.plan(bridge,'left')
        bridge.executing_arm='left'
        right=bridge.robots['right']
        delta=bridge.guard.cfg['max_peer_joint_drift_rad']*.65
        right.angles[right.names[0]]+=delta
        bridge.check_other_arm_path('left',[None])
        accepted=bridge.peer_reference
        self.assertAlmostEqual(accepted['joint_rad'][0],delta)
        right.angles[right.names[0]]+=delta
        bridge.check_running_arm('left')
        changed=right.read_state()
        changed['joint_rad'][0]+=5*delta
        with patch.object(right,'read_state',side_effect=[right.read_state(),changed]):
            with self.assertRaises(PeerStateChangedError):
                bridge.check_other_arm_path('left',[None])
        self.assertIs(bridge.peer_reference,accepted)

    def test_peer_move_during_execute_validation_returns_a_consumed_zero_command_plan(self):
        bridge=self.bridge(True)
        planned=self.plan(bridge,'left')
        right=bridge.robots['right']
        stable=right.read_state()
        changed=dict(stable,joint_rad=[.04]+stable['joint_rad'][1:])
        with patch.object(right,'read_state',side_effect=[stable,changed,changed]):
            receipt=bridge.execute('left',planned['plan_id'])
        self.assertEqual(receipt['failed_stage']['phase'],'dispatch_validation')
        self.assertFalse(receipt['outcome_uncertain'])
        self.assertFalse(receipt['stop_attempted'])
        self.assertEqual(receipt['confirmed_commands'],0)
        self.assertIsNone(bridge.services['left'].prepared)
        self.assertEqual(bridge.shared.remaining,12)
        bridge.robots['left'].grip.command_position.assert_not_called()

    def test_peer_change_during_motion_stops_active_arm_and_never_drives_peer(self):
        bridge=self.bridge(True);left=bridge.robots['left'];right=bridge.robots['right']
        prepared=self.plan(bridge,'left','move_to',[.65,.2,.1])
        def interrupted(*args,**kwargs):
            right.angles[right.names[0]]+=.02
            left.ready()
        left.move_segment=Mock(side_effect=interrupted)
        receipt=bridge.execute('left',prepared['plan_id'])
        self.assertEqual(receipt['failure_type'],'PeerJointMovedError')
        self.assertTrue(receipt['outcome_uncertain']);self.assertFalse(receipt['retryable'])
        self.assertEqual(receipt['selected_arm'],'left');self.assertEqual(receipt['status'],'failed')
        left.limb.set_joint_positions.assert_called_once();left.grip.stop.assert_called_once()
        right.limb.set_joint_positions.assert_not_called();right.grip.command_position.assert_not_called()
        for service in bridge.services.values():self.assertIsNone(service.prepared)

    def test_collision_rejection_precedes_gripper_and_consumes_observation(self):
        bridge=self.bridge(True)
        bridge.guard.models['right'].capsules=lambda *a,**k:[Capsule('right_forearm',(.5,-1,0),(.5,1,0),.03)]
        result=self.plan(bridge,'left')
        self.assertEqual(result['failure_code'],'inter_arm_clearance');self.assertFalse(result['applied'])
        self.assertIsNone(bridge.services['left'].prepared);self.assertIsNone(bridge.services['left'].observation)
        for robot in bridge.robots.values():robot.grip.command_position.assert_not_called()

    def test_concurrent_direct_observe_waits_for_execute_to_finish(self):
        bridge=self.bridge(True);prepared=self.plan(bridge,'left')
        entered=threading.Event();release=threading.Event();observed=threading.Event();errors=[]
        def grip(*args,**kwargs):entered.set();release.wait(2);return True
        bridge.robots['left'].grip.command_position.side_effect=grip
        def execute():
            try:bridge.execute('left',prepared['plan_id'])
            except Exception as error:errors.append(error)
        def observe():
            try:bridge.observe('right');observed.set()
            except Exception as error:errors.append(error)
        execution=threading.Thread(target=execute);execution.start();self.assertTrue(entered.wait(1))
        observation=threading.Thread(target=observe);observation.start()
        self.assertFalse(observed.wait(.05))
        release.set();execution.join(2);observation.join(2)
        self.assertFalse(errors);self.assertTrue(observed.is_set())


class CapsuleGeometryTests(unittest.TestCase):
    def test_metric_swept_margin_can_cover_measured_peer_drift_but_never_grip_change(self):
        cfg=configuration()['inter_arm_clearance']
        cfg.update(peer_motion_policy='swept_clearance',max_peer_joint_drift_rad=.01)
        models={name:FakeModel(name,cfg) for name in ('left','right')}
        guard=InterArmGuard(cfg,models)
        models['right'].capsule_displacement_bounds=lambda excess:{'wrist':2*excess['right_j0']}
        ref={'joint_names':['right_j0'],'joint_rad':[0.],'gripping':False,'gripper_percent':100.}
        sample={**ref,'joint_rad':[.05]}
        checked={'checked':True,'swept_intervals_checked':True,'minimum_clearance_m':.15}
        self.assertAlmostEqual(guard.check_peer(ref,sample,checked_path=checked,peer_arm='right')['right_j0'],.05)
        with self.assertRaises(PeerStateChangedError):
            guard.check_peer(ref,sample,checked_path={**checked,'minimum_clearance_m':.10},peer_arm='right')
        with self.assertRaisesRegex(PeerStateChangedError,'occupancy'):
            guard.check_peer(ref,{**sample,'gripping':True},checked_path=checked,peer_arm='right')
        with self.assertRaises(PeerStateChangedError):
            guard.check_peer(ref,sample,peer_arm='right')

    def test_middle_link_crossing_is_detected_even_with_far_apart_endpoints(self):
        left=Capsule('left_elbow',(-1,0,0),(1,0,0),.05)
        right=Capsule('right_forearm',(0,-1,0),(0,1,0),.05)
        self.assertGreater(sum((a-b)**2 for a,b in zip(left.end,right.end)),1.)
        with self.assertRaises(ArmCollisionError):check_capsules([left],[right],.04)
    def test_parallel_degenerate_and_skew_segments(self):
        self.assertAlmostEqual(segment_distance([0,0,0],[1,0,0],[0,2,0],[1,2,0]),2.)
        self.assertAlmostEqual(segment_distance([0,0,0],[0,0,0],[1,0,0],[2,0,0]),1.)
        self.assertAlmostEqual(segment_distance([-1,0,0],[1,0,0],[0,-1,2],[0,1,2]),2.)
    def test_joint_path_checks_intermediate_full_link_configurations(self):
        cfg=configuration()['inter_arm_clearance'];models={arm:FakeModel(arm,cfg) for arm in ('left','right')}
        seen=[]
        def active(joints,holding):
            q=joints['left_j0'];seen.append(q)
            return [Capsule('left_elbow',(0,.4-q,0),(1,.4-q,0),.03)]
        models['left'].capsules=active
        guard=InterArmGuard(cfg,models)
        peer={'joint_names':['right_j0'],'joint_rad':[0.],'gripping':False,'gripper_percent':100.}
        with self.assertRaises(ArmCollisionError):
            guard.path_clearance('left',{'left_j0':0.},[[({'left_j0':1.6},None)]],peer,peer)
        self.assertTrue(any(0.<q<1.6 for q in seen))
    def test_swept_padding_rejects_a_gap_smaller_than_motion_bound(self):
        cfg=configuration()['inter_arm_clearance'];models={arm:FakeModel(arm,cfg) for arm in ('left','right')}
        models['left'].capsules=lambda *a:[Capsule('left_elbow',(0,0,0),(1,0,0),.03)]
        models['right'].capsules=lambda *a:[Capsule('right_elbow',(0,.102,0),(1,.102,0),.03)]
        guard=InterArmGuard(cfg,models)
        peer={'joint_names':['right_j0'],'joint_rad':[0.],'gripping':False,'gripper_percent':100.}
        self.assertGreater(check_capsules(models['left'].capsules({}),models['right'].capsules({}),.04),.04)
        with self.assertRaises(ArmCollisionError):guard.path_clearance('left',{'left_j0':0.},[[({'left_j0':.02},None)]],peer,peer)
    def test_peer_drift_envelope_is_anchored_at_observation_not_later_shifted_pose(self):
        cfg=configuration()['inter_arm_clearance'];models={arm:FakeModel(arm,cfg) for arm in ('left','right')}
        original=models['right'].capsules;seen=[]
        def peer_geometry(joints,holding):seen.append(dict(joints));return original(joints,holding)
        models['right'].capsules=peer_geometry
        guard=InterArmGuard(cfg,models)
        baseline={'joint_names':['right_j0'],'joint_rad':[0.],'gripping':False,'gripper_percent':100.}
        current={**baseline,'joint_rad':[.0005]}
        guard.path_clearance('left',{'left_j0':0.},[],baseline,current)
        self.assertEqual(seen,[{'right_j0':0.}])
    def test_missing_link_radii_or_tool_dimensions_are_not_silently_defaulted(self):
        cfg=configuration()['inter_arm_clearance'];cfg['link_radii_m']['right']={}
        with self.assertRaisesRegex(ValueError,'Explicit radii'):validate_clearance_config(cfg)
        cfg=configuration()['inter_arm_clearance'];cfg['tool_capsules']['right']=[]
        with self.assertRaisesRegex(ValueError,'actual gripper'):validate_clearance_config(cfg)


def synthetic_urdf():
    joints=[]
    for arm in ('left','right'):
        previous='base'
        for index,(child,kind) in enumerate([(arm+'_arm_mount','fixed')]+[(arm+'_link_'+str(i),'revolute') for i in range(7)]+[(arm+'_gripper','fixed')]):
            joints.append('<joint name="{}_j{}" type="{}"><parent link="{}"/><child link="{}"/></joint>'.format(arm,index,kind,previous,child))
            previous=child
    return '<robot name="offline-baxter">'+''.join(joints)+'</robot>'


class CollisionModelTests(unittest.TestCase):
    def model_dependencies(self,cfg,body=False):
        class Vector:
            def __init__(self,x=0,y=0,z=0):self.v=(x,y,z)
            def __getitem__(self,i):return self.v[i]
            def Norm(self):return sum(x*x for x in self.v)**.5
        class Frame:
            def __init__(self,rotation=None,position=None):self.p=position or Vector()
            def __mul__(self,value):return Vector(*(self.p[i]+value[i] for i in range(3)))
        links=['left_arm_mount']+arm_chain_links(synthetic_urdf(),'left')
        segments=[]
        for index,name in enumerate(links):
            kind=1 if 1<=index<=7 else 0;jointname='left_j'+str(index)
            joint=NS(getType=lambda kind=kind:kind,getName=lambda name=jointname:name)
            tip=Frame();tip.p=Vector(.1,0,0)
            segments.append(NS(getName=lambda name=name:name,getJoint=lambda joint=joint:joint,getFrameToTip=lambda tip=tip:tip))
        chain=NS(getNrOfSegments=lambda:len(segments),getSegment=lambda i:segments[i])
        calls=[]
        class FK:
            def __init__(self,chain):self.chain=chain
            def JntToCart(self,q,frame,segment=None):
                segment=self.chain.getNrOfSegments() if segment is None else segment
                calls.append(segment);frame.p=Vector(.1*segment,0,0);return 0
        joint_type=NS();setattr(joint_type,'None',0)
        kdl=NS(Joint=joint_type,JntArray=lambda n:[0.]*n,Frame=Frame,Vector=Vector,ChainFkSolverPos_recursive=FK,
               Rotation=NS(RPY=lambda *args:None))
        robot=NS(joint_map={'left_j'+str(i):NS(limit=NS(lower=-2.,upper=2.)) for i in range(1,8)})
        class Cylinder:
            length=.2;radius=.06
        def get_chain(base,link):
            if not body or link=='left_gripper':return chain
            count=links.index(link)+1 if link in links else 5
            return NS(getNrOfSegments=lambda:count,getSegment=lambda i:segments[i])
        if body:
            robot.links=[NS(name=name,collisions=[NS(geometry=Cylinder(),origin=NS(xyz=[0,0,.05],rpy=[0,0,0]))])
                         for name in links[1:]+['left_fixed_visual']]
        modules={'PyKDL':kdl,'urdf_parser_py.urdf':NS(URDF=NS(from_parameter_server=lambda **kw:robot),Cylinder=Cylinder),
                 'cix_harness.upstream.kdl_parser':NS(kdl_tree_from_urdf_model=lambda robot:NS(getChain=get_chain))}
        return modules,calls

    def test_template_uses_actual_chain_names_and_never_verifies_or_arms(self):
        clearance=build_collision_candidate(synthetic_urdf())
        self.assertEqual(list(clearance['link_radii_m']['left']),['left_link_'+str(i) for i in range(7)]+['left_gripper'])
        self.assertFalse(clearance['verified']);self.assertIn('UNMEASURED',clearance['geometry_source'])
        cfg=configuration(True)
        combined=build_dual_candidate(cfg['arms']['left'],cfg['arms']['right'],clearance=clearance)
        for arm in ('left','right'):
            self.assertFalse(combined['arms'][arm]['commissioned'])
            self.assertFalse(combined['arms'][arm]['primitives']['calibrated'])
        self.assertFalse(combined['inter_arm_clearance']['verified'])

    def test_kdl_uses_every_link_frame_and_explicit_installed_tool_not_just_endpoint(self):
        cfg=build_collision_candidate(synthetic_urdf());modules,calls=self.model_dependencies(cfg)
        with patch.dict('sys.modules',modules):
            model=KDLArmCapsules('left',cfg)
            capsules=model.capsules({name:0. for name in model.names},holding=True)
        self.assertEqual(calls,list(range(1,10)))
        self.assertEqual([c.name for c in capsules[:-1]],arm_chain_links(synthetic_urdf(),'left'))
        self.assertAlmostEqual(capsules[0].start[0],.1)
        self.assertAlmostEqual(capsules[0].end[0],.2)
        self.assertAlmostEqual(capsules[-1].end[0],.9)
        self.assertAlmostEqual(capsules[-1].end[2],.2)
        self.assertAlmostEqual(capsules[-1].radius,.16)
        padding=model.capsule_displacement_bounds({n:.015 for n in model.names})
        self.assertEqual(padding['left_link_0'],0.)
        self.assertGreater(padding['left_installed_tool_0'],padding['left_link_1'])

    def test_static_shoulder_does_not_inherit_wrist_motion_padding(self):
        left=[Capsule('left_shoulder',(0,0,0),(0,0,0),.10),Capsule('left_tool',(1,0,0),(1,0,0),.05)]
        right=[Capsule('right_shoulder',(0,.25,0),(0,.25,0),.10),Capsule('right_tool',(1,.5,0),(1,.5,0),.05)]
        distance=check_capsules(left,right,.04,active_padding={'left_shoulder':0.,'left_tool':.10},
                                peer_padding={'right_shoulder':0.,'right_tool':.10})
        self.assertAlmostEqual(distance,.05)
        with self.assertRaises(ArmCollisionError):
            check_capsules(left,right,.04,active_padding={'left_shoulder':.02,'left_tool':0.},
                           peer_padding={'right_shoulder':0.,'right_tool':0.})

    def test_urdf_cylinders_include_fixed_branches_offsets_and_own_joint_rotation(self):
        cfg=build_collision_candidate(synthetic_urdf());cfg.update(body_geometry='urdf_cylinders',urdf_padding_m=.02)
        modules,_=self.model_dependencies(cfg,body=True)
        with patch.dict('sys.modules',modules):
            model=KDLArmCapsules('left',cfg)
            capsules=model.capsules({name:0. for name in model.names})
        bodies={c.name:c for c in capsules}
        shoulder=bodies['left_link_0__collision_0']
        self.assertAlmostEqual(shoulder.radius,.08)
        self.assertAlmostEqual(shoulder.start[2],-.05)
        self.assertAlmostEqual(shoulder.end[2],.15)
        self.assertIn('left_fixed_visual__collision_0',bodies)
        self.assertIn('left_installed_tool_0',bodies)
        deltas={n:0. for n in model.names};deltas['left_j1']=.1
        padding=model.capsule_displacement_bounds(deltas)
        # Finite-cylinder support can rotate with the joint. Bound the
        # outermost radial surface as well as the endpoints of its axis.
        self.assertAlmostEqual(padding[shoulder.name],(.15+.06)*.1)
        self.assertEqual(set(padding),set(bodies))

    def test_certified_finite_end_face_keeps_real_padding_and_separation(self):
        arm=[Capsule('forearm',(0,0,-.05),(0,0,.05),.08,.02)]
        peer=lambda z:[Capsule('mount',(0,0,z),(0,0,z),.02)]
        self.assertAlmostEqual(check_capsules(arm,peer(.15),.04),.06)
        with self.assertRaises(ArmCollisionError):check_capsules(arm,peer(.125),.04)
        with self.assertRaises(ArmCollisionError):
            check_capsules(arm,peer(.15),.04,peer_padding={'mount':.021})

    def test_urdf_mode_rejects_uncovered_main_link_or_unsupported_shape(self):
        cfg=build_collision_candidate(synthetic_urdf());cfg.update(body_geometry='urdf_cylinders',urdf_padding_m=.02)
        for missing in (True,False):
            modules,_=self.model_dependencies(cfg,body=True)
            robot=modules['urdf_parser_py.urdf'].URDF.from_parameter_server()
            if missing:robot.links[0].collisions=[]
            else:robot.links[0].collisions[0].geometry=NS(mesh='unknown')
            with patch.dict('sys.modules',modules),self.assertRaises(ClearanceUnavailableError):
                KDLArmCapsules('left',cfg)

    def test_descendant_lever_bound_covers_compound_rotation_without_static_base_padding(self):
        import math
        model=object.__new__(KDLArmCapsules)
        model.names=['shoulder','elbow'];model.joint_reach_bounds={'shoulder':1.6,'elbow':.6}
        def endpoint(a,b):return (10+math.cos(a)+.6*math.cos(a+b),math.sin(a)+.6*math.sin(a+b))
        for a,b,da,db in [(0,0,.015,.015),(1,-.7,-.02,.03),(-1,2,.01,-.02)]:
            bound=model.displacement_bound({'shoulder':da,'elbow':db})
            self.assertLessEqual(math.dist(endpoint(a,b),endpoint(a+da,b+db)),bound)
            self.assertLess(bound,(abs(da)+abs(db))*11.6)
        with self.assertRaises(ClearanceUnavailableError):model.displacement_bound({'shoulder':.01})

    def test_kdl_missing_actual_link_radius_fails_closed(self):
        cfg=build_collision_candidate(synthetic_urdf());modules,_=self.model_dependencies(cfg)
        del cfg['link_radii_m']['left']['left_link_3']
        with patch.dict('sys.modules',modules),self.assertRaisesRegex(ClearanceUnavailableError,'exactly these'):
            KDLArmCapsules('left',cfg)


if __name__=='__main__':unittest.main()
