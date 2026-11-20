import copy
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch

from cix_harness import ring_peg
from cix_harness.primitive_service import PrimitiveService
from cix_harness.primitives import plan
from cix_harness.task_plan import workspace_context
from cix_harness.harness import run
from tests import test_primitives
from tests.test_executive import Journal,Robot,ScriptedPlanner,step,check
from tests.test_primitives import profile


def example():
    p=profile();p.update(table_z_m=0.,grasp_inset_m=.003,tcp_offset_tool_m=[0.,0.,0.],
                          approach_path='direct_hover',tool_orientation_xyzw=[0,0,0,1])
    p['ring_peg']={'perception':{},'ring_thickness_m':.010,'horizontal_uncertainty_m':.001,
        'vertical_uncertainty_m':.0008,'jaw_axis_tool':[1,0,0],'jaw_open_half_span_m':.025,
        'finger_half_width_m':.002,'fingertip_below_tcp_m':.001,'release_gap_m':.003,'max_drop_m':.065}
    ring={'id':'red-ring','color':'red','location':'loose','peg_id':None,'center_uv_px':[50,40],
          'rim_grasp_uv_px':[54,40],'center_base_m':[.7,.2,.04],'rim_grasp_base_m':[.715,.2,.04],
          'inner_radius_m':.010,'outer_radius_m':.020,'normal_base':[0,0,1]}
    peg={'id':'peg-red','color':'red','tip_uv_px':[20,45],'tip_base_m':[.74,.27,.09],
         'axis_base':[0,0,1],'radius_m':.004,'support_z_m':.05,'stack_count':1}
    scene={'obs_id':'fresh','resolution':[101,101],'rings':[ring],'pegs':[peg],'held_ring_id':None}
    state={'eef_m':[.710,.198,.035],'eef_xyzw':[0,0,0,1],'gripping':False}
    return p,scene,state,ring,peg


class RingTask(unittest.TestCase):
    def test_wrist_joins_stationary_ros_header_within_50ms_arrival_skew_even_at_low_rate(self):
        p,scene,state,ring,peg=example();state.update(gripping=True,arm='left');ring['location']='held';scene['held_ring_id']=ring['id']
        p['ring_peg']['wrist_alignment']={'wait_for_frame_s':.1};stamp=lambda x:NS(to_sec=lambda:x)
        msg=NS(header=NS(seq=73,frame_id='/left_hand_camera',stamp=stamp(925.45)))
        robot=NS(wrist_condition=threading.Condition(),wrist_state=(10.97,msg),ros=NS(Time=NS(now=lambda:stamp(926.))))
        service=PrimitiveService.__new__(PrimitiveService);service.profile=p;service.robot=robot
        converter=Mock();converter.imgmsg_to_cv2.return_value='pixels'
        with patch.dict('sys.modules',{'cv_bridge':NS(CvBridge=lambda:converter)}),patch('cix_harness.primitive_service.time.monotonic',return_value=11.),patch('cix_harness.ring_wrist.measure',return_value={'offset_xy_m':[.001,.002]}):
            result=service._ring_wrist_alignment(scene,state,peg['id'],10.99,10.30)
            self.assertEqual((result['frame_number'],result['frame_id']),(73,'/left_hand_camera'));self.assertAlmostEqual(result['source_age_s'],.55)
            converter.imgmsg_to_cv2.assert_called_once_with(msg,desired_encoding='bgr8')
            with self.assertRaises(ring_peg.RingGeometryError):service._ring_wrist_alignment(scene,state,peg['id'],10.99,10.80)
            msg.header.frame_id='/right_hand_camera'
            with self.assertRaises(ring_peg.RingGeometryError):service._ring_wrist_alignment(scene,state,peg['id'],10.99,10.30)
    def test_harness_replans_a_visible_miss_with_empty_hand_then_verifies_new_place(self):
        class RingRobot(Robot):
            def call(self,method,args=(),**kwargs):
                result=super().call(method,args,**kwargs)
                if method=='primitive.observe':
                    miss={'ring_id':'blue','color':'blue','peg_id':'p1','center_uv_px':[150,170],'obs_id':str(self.i)}
                    result['ring_scene']={'verified_skill':{2:'pick',4:'pick',5:'place'}.get(self.i),'missed_ring':miss if self.i==3 else None}
                return result
        with tempfile.TemporaryDirectory() as folder:
            args=NS(task='将散落蓝环套到蓝杆',route='primitives',execute=True,vlm_port=9110,vlm_model='replay',robot_port=9102,
                    task_kind='place',max_turns=6,max_recoveries=2,max_grasp_retries=1,max_replans=2,max_planner_calls=20,memory=str(Path(folder)/'memory'))
            steps=[step('pick','blue ring'),step('place','blue peg')];decisions=[check(uv=[.4,.25]),check('step_done'),check(uv=[.7,.6]),
                check(uv=[.45,.36]),check('step_done'),check(uv=[.7,.6]),check('step_done'),check('task_done')]
            planner=ScriptedPlanner([steps,steps],decisions);robot=RingRobot();journal=Journal(folder)
            self.assertEqual(run(args,journal,robot,planner),0);self.assertEqual(robot.skills,['pick','place','pick','place'])
            self.assertEqual(len([e for name,e in journal.events if name=='ring_missed_peg']),1)
            self.assertEqual([e['step']['skill'] for name,e in journal.events if name=='task_step_verified'],['pick','pick','place'])
    def test_visual_servo_moves_once_from_fresh_relative_error_then_checks_next_frame_before_open(self):
        p,scene,state,ring,peg=example();p['calibrated']=True
        ring.update(location='held',center_base_m=[.752,.27,.106]);scene['held_ring_id']=ring['id']
        state.update(gripping=True,eef_m=[.752,.27,.105],joint_names=['j'],joint_rad=[0.])
        stages=[{'kind':'gripper','open':True},{'kind':'verify_grasp','held':False}]
        s,r=test_primitives.Execution().service(stages,True);s.profile=p;r.cfg['primitives']=p
        token,created,initial,ss,paths,n,_,_=s.prepared;s.prepared=(token,created,initial,ss,paths,n,'place',True)
        _,detail=ring_peg.release_geometry(p,scene,state,ring,peg);s.ring_planned=(token,dict(detail,skill='place',obs_id='initial'))
        r.read_state.side_effect=lambda:copy.deepcopy(state);r.endpoint_position.side_effect=lambda:list(state['eef_m'])
        observations=[];r.camera.capture.side_effect=lambda:('',time.monotonic(),len(observations)+1)
        def observe(*args):
            observations.append(copy.deepcopy(scene));return dict(copy.deepcopy(scene),frame_number=len(observations))
        def preflight(suffix,*args,**kwargs):
            self.assertEqual(suffix[1:],stages[:2]);return initial,[[(initial,suffix[0]['tcp_m'])],None,None]
        def move(joints,path,*args,**kwargs):
            new=path[-1][1];ring['center_base_m']=[x+y-z for x,y,z in zip(ring['center_base_m'],new,state['eef_m'])];state['eef_m']=new
        def opened(*args,**kwargs):
            self.assertEqual(len(observations),2);state['gripping']=False;return True
        r.move_segment.side_effect=move;r.grip.gripping.side_effect=lambda:state['gripping'];r.grip.command_position.side_effect=opened
        with patch.object(s,'_ring_scene',side_effect=observe),patch.object(s,'_preflight',side_effect=preflight) as check_path:
            receipt=s.execute(token)
        self.assertEqual(receipt['status'],'completed',receipt);self.assertEqual(check_path.call_count,1)
        self.assertEqual(r.move_segment.call_count,1);self.assertEqual(r.grip.command_position.call_count,1)
        self.assertAlmostEqual(state['eef_m'][0],.744);self.assertEqual(r.motion_remaining,18)
        self.assertEqual([x['stage']['kind'] for x in receipt['completed_stages']],['move','gripper','verify_grasp'])

    def test_outer_grip_uses_wall_height_and_cannot_pick_colour_marker(self):
        p,scene,state,ring,peg=example();selected,_=ring_peg.resolve(p,scene,state,'pick',[.50,.40])
        c,e=ring_peg.pick_geometry(p,scene,selected,state['eef_xyzw']);self.assertEqual(e['grasp_mode'],'outer')
        self.assertEqual(c[:2],ring['center_base_m'][:2]);self.assertAlmostEqual(c[2],.037)
        segments=plan('pick',state,p,ring['rim_grasp_base_m'],ring_contact_tcp=c)
        close=segments.index({'kind':'gripper','open':False});self.assertEqual(segments[close-1]['tcp_m'],c)
        for location in ('on_peg','unknown',None):
            ring['location']=location
            with self.assertRaises(ring_peg.RingGeometryError):ring_peg.resolve(p,scene,state,'pick',[.50,.4])
        ring['location']='loose';peg['color']='blue'
        with self.assertRaises(ring_peg.RingGeometryError):ring_peg.resolve(p,scene,state,'pick',[.50,.4])

    def test_rim_grip_checks_open_finger_clearance_direction_and_depth(self):
        p,scene,state,ring,peg=example();s=p['ring_peg'];s.update(grasp_mode='rim',jaw_open_half_span_m=.009)
        c,e=ring_peg.pick_geometry(p,scene,ring,[0,0,0,1]);self.assertEqual(c[:2],ring['rim_grasp_base_m'][:2])
        with self.assertRaises(ring_peg.RingGeometryError):ring_peg.pick_geometry(p,scene,ring,[0,0,.70710678,.70710678])
        s['fingertip_below_tcp_m']=.008
        _,shallow=ring_peg.pick_geometry(p,scene,ring,[0,0,0,1]);self.assertLess(shallow['effective_grasp_inset_m'],.001)
        s['fingertip_below_tcp_m']=.010
        with self.assertRaises(ring_peg.RingGeometryError):ring_peg.pick_geometry(p,scene,ring,[0,0,0,1])

    def test_fresh_local_table_upper_bound_sets_pick_depth_for_thin_ring(self):
        p,scene,state,ring,peg=example();p.update(grasp_inset_m=.010);p['ring_peg'].update(vertical_uncertainty_m=.012,tcp_table_error_m=.0007)
        ring.update(local_table_z_m=.030,local_table_upper_z_m=.034,local_table_depth_count=504)
        contact,e=ring_peg.pick_geometry(p,scene,ring,state['eef_xyzw']);self.assertAlmostEqual(contact[2],.0362)
        self.assertLess(e['effective_grasp_inset_m'],.004);self.assertEqual(e['local_table_upper_z_m'],.034)

    def test_offset_ring_requires_measured_hole_peg_and_stack_before_open(self):
        p,scene,state,ring,peg=example();ring['location']='held';state['gripping']=True;scene['held_ring_id']=ring['id']
        ring_peg.resolve(p,scene,state,'place',[.20,.45]);c,e=ring_peg.release_geometry(p,scene,state,ring,peg)
        self.assertAlmostEqual(c[0],.750);self.assertAlmostEqual(c[1],.268)
        self.assertGreater(c[2],peg['tip_base_m'][2]);self.assertGreater(e['free_xy_m'],0)
        for i in range(3):ring['center_base_m'][i]+=c[i]-state['eef_m'][i]
        state['eef_m']=c
        measured=ring_peg.verify_before_open(p,scene,state,e);self.assertAlmostEqual(measured['release_xy_error_m'],0)
        ring['center_base_m'][0]+=.011
        with self.assertRaises(ring_peg.RingGeometryError):ring_peg.verify_before_open(p,scene,state,e)
        ring['center_base_m'][0]-=.011;peg['color']='blue'
        with self.assertRaises(ring_peg.RingGeometryError):ring_peg.verify_before_open(p,scene,state,e)
        peg['color']='red';peg['radius_m']=.012
        with self.assertRaises(ring_peg.RingGeometryError):ring_peg.release_geometry(p,scene,state,ring,peg)
        peg['radius_m']=.004;peg['support_z_m']=None
        with self.assertRaises(ring_peg.RingGeometryError):ring_peg.release_geometry(p,scene,state,ring,peg)

    def test_existing_ring_on_peg_cannot_verify_a_new_drop(self):
        p,scene,state,ring,peg=example();ring['location']='held';scene['held_ring_id']=ring['id'];state['gripping']=True
        _,e=ring_peg.release_geometry(p,scene,state,ring,peg);e.update(skill='place',obs_id='before')
        state['gripping']=False
        self.assertIsNone(ring_peg.verified(e,scene,state));peg['stack_count']=2
        self.assertEqual(ring_peg.verified(e,scene,state),'place');scene['obs_id']='before'
        self.assertIsNone(ring_peg.verified(e,scene,state))

    def test_stack_depth_difference_requires_a_matching_paired_reference_and_no_loose_ring(self):
        p,scene,state,ring,peg=example();ring['location']='held';state['gripping']=True
        peg.update(stack_count=None,stack_delta_from_frame=12,stack_depth_delta_m=-.001)
        _,e=ring_peg.release_geometry(p,scene,state,ring,peg);e.update(skill='place',obs_id='old')
        state['gripping']=False;peg.update(stack_depth_delta_m=.009,stack_rise_supported=True)
        self.assertEqual(ring_peg.verified(e,scene,state),'place');ring['location']='loose'
        self.assertIsNone(ring_peg.verified(e,scene,state));ring['location']='unknown';peg['stack_delta_from_frame']=13
        self.assertIsNone(ring_peg.verified(e,scene,state))

    def test_manually_matched_toy_uses_centered_gravity_attempt_then_recognizes_visible_miss(self):
        p,scene,state,ring,peg=example();p['ring_peg']['matching_parts_verified']=True
        ring.update(location='held',center_base_m=[.74,.27,.106],inner_radius_m=.0041);peg['radius_m']=.00445
        state.update(gripping=True,eef_m=[.74,.27,.105]);scene['held_ring_id']=ring['id']
        _,e=ring_peg.release_geometry(p,scene,state,ring,peg);e.update(skill='place',obs_id='before')
        self.assertLess(e['free_xy_m'],0);now=ring_peg.verify_before_open(p,scene,state,e)
        self.assertEqual(now['release_basis'],'centered_matching_parts_gravity_attempt')
        ring['center_base_m'][0]+=.0015
        with self.assertRaises(ring_peg.RingAlignmentNeeded):ring_peg.verify_before_open(p,scene,state,e)
        self.assertEqual(ring_peg.verify_before_open(p,scene,state,e,servo_budget_spent=True)['release_basis'],now['release_basis'])
        ring['center_base_m'][0]+=.003
        with self.assertRaises(ring_peg.RingAlignmentNeeded):ring_peg.verify_before_open(p,scene,state,e,servo_budget_spent=True)
        state['gripping']=False;ring.update(location='loose',peg_id=None)
        self.assertEqual(ring_peg.missed_ring(e,scene,state)['ring_id'],ring['id']);self.assertIsNone(ring_peg.verified(e,scene,state))

    def test_service_plans_from_observed_shape_and_rejects_stale_or_direct_xyz(self):
        import numpy as np
        p,scene,state,ring,peg=example();robot=Mock();robot.cfg={'primitives':p,'max_observation_age_s':30,
            'planning':{'primitive_skills':['pick','place'],'grasp_yaw_offsets_rad':[0.]}};robot.motion_remaining=100
        robot.joints.return_value={'left_w0':0.};service=PrimitiveService(robot)
        def mock_frame():service.observation=({'id':'fresh','state':state,'ring_scene':copy.deepcopy(scene)},time.monotonic(),np.zeros((101,101)))
        mock_frame()
        with patch.object(service,'_preflight',return_value=({'left_w0':0.},[])),patch('cix_harness.primitive_service.ground_pixel',side_effect=AssertionError('hole depth used')):
            result=service.prepare('fresh','pick',[.5,.4]);self.assertEqual(result['ring_geometry']['grasp_mode'],'outer')
            self.assertEqual(result['ring_geometry']['obs_id'],'fresh')
        mock_frame();self.assertEqual(service.prepare('fresh','pick',target_base_m=[.7,.2,.03])['failure_code'],'ring_geometry')
        scene['obs_id']='other';mock_frame();self.assertEqual(service.prepare('fresh','pick',[.5,.4])['failure_code'],'ring_geometry')
        self.assertEqual(workspace_context({'ring_scene':scene},{})['ring_scene'],scene)
