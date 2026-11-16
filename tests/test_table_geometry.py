"""Synthetic sloped-table and long-path checks, without ROS or hardware access."""
import copy
import json
import math
import time
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock,patch

from cix_harness.kinematics import IKUnreachableError
from cix_harness.prepare_site import build_candidate_profiles
from cix_harness.primitives import (TargetWorkspaceError,_slerp,check_tcp,eef_from_tcp,
    plan,table_height,validate_profile,workspace_polygon)
from cix_harness.primitive_service import PrimitiveService
from cix_harness.workspace import workspace_metadata


def profile():
    p=json.loads(Path('config/baxter-primitives.example.json').read_text())['primitives']
    p.update(table_plane={'normal':[-.1,0,1],'offset':.25},
             table_polygon_base_xy=[[.4,-.5],[1.,-.4],[.9,.5],[.35,.4]],max_leg_m=.12)
    p['tcp_workspace_min_m'][2]=-.25
    return p


def state(held=False):
    return {'eef_m':[.45,-.3,.1],'eef_xyzw':[1,0,0,0],'gripping':held,
            'joint_names':['j'],'joint_rad':[0.]}


def service(p=None,held=False,robot_type=Mock):
    robot=robot_type();robot.cfg={'primitives':p or profile(),'max_observation_age_s':3}
    robot.motion_remaining=100;robot.joints.return_value={'j':0.}
    st=state(held)
    robot.limb.endpoint_pose.return_value={'position':NS(**dict(zip(('x','y','z'),st['eef_m']))),
                                          'orientation':NS(x=1.,y=0.,z=0.,w=0.)}
    robot.continuous_ik.solve.return_value={'j':0.}
    instance=PrimitiveService(robot)
    instance.observation=({'id':'o','state':st},time.monotonic(),None)
    modules={'geometry_msgs.msg':NS(PoseStamped=lambda:NS(header=NS(),pose=NS(position=NS(),orientation=NS()))),
             'tf':NS(transformations=NS(quaternion_slerp=_slerp))}
    return instance,robot,modules


class TableGeometry(unittest.TestCase):
    def test_plane_replaces_flat_height_and_is_scale_invariant(self):
        p=profile()
        self.assertAlmostEqual(table_height(.4,-.3,p),-.21)
        check_tcp([.4,-.3,-.195],p)  # Valid on lower side; old flat guard would reject.
        with self.assertRaisesRegex(ValueError,'below table'):
            check_tcp([.9,.2,-.18],p)  # Invalid on higher side; old flat guard allowed it.
        p['table_plane']={'normal':[-.2,0,2],'offset':.5}
        self.assertAlmostEqual(table_height(.4,-.3,p),-.21)
        del p['table_z_m'];validate_profile(p)

    def test_invalid_planes_and_nonconvex_boundaries_fail_closed(self):
        for plane in ({'normal':[0,0,0],'offset':0},{'normal':[0,0,-1],'offset':0},
                      {'normal':[1,0,.01],'offset':0},{'normal':[0,0,1],'offset':float('nan')}):
            with self.subTest(plane=plane),self.assertRaises(ValueError):
                p=profile();p['table_plane']=plane;validate_profile(p)
        polygons=[[[.4,0],[.8,0],[.6,.1],[.8,.4],[.4,.4]],
                  [[.4,0],[.8,.4],[.4,.4],[.8,0]],[[.4,0],[.8,0],[.4,0]],
                  [[2,0],[3,0],[3,1],[2,1]]]
        for polygon in polygons:
            with self.subTest(polygon=polygon),self.assertRaises(ValueError):
                p=profile();p['table_polygon_base_xy']=polygon;validate_profile(p)

    def test_polygon_both_windings_and_target_vs_path_error(self):
        p=profile()
        for polygon in (p['table_polygon_base_xy'],list(reversed(p['table_polygon_base_xy']))):
            p['table_polygon_base_xy']=polygon;check_tcp([.7,.2,.1],p)
            with self.assertRaises(TargetWorkspaceError):check_tcp([.36,-.6,.1],p,selected_target=True)
            with self.assertRaises(ValueError) as raised:check_tcp([.36,-.6,.1],p)
            self.assertNotIsInstance(raised.exception,TargetWorkspaceError)
            with self.assertRaisesRegex(ValueError,'below table') as raised:
                check_tcp([.36,-.6,-.4],p,selected_target=True)
            self.assertNotIsInstance(raised.exception,TargetWorkspaceError)

    def test_plane_clearance_includes_held_object(self):
        p=profile();check_tcp([.9,.2,-.14],p)
        with self.assertRaisesRegex(ValueError,'held object'):
            check_tcp([.9,.2,-.14],p,held_bottom_m=.02)

    def test_projection_uses_clipped_polygon_and_per_vertex_plane_height(self):
        p=profile();p['tcp_workspace_min_m'][0]=.5;p['tcp_workspace_max_m'][1]=.3
        p['camera_to_base_translation_m']=[0,0,-1];before=copy.deepcopy(p)
        intrinsics=NS(width=101,height=101)
        result=workspace_metadata(p,intrinsics,lambda _,v:[50+10*v[0]/v[2],50+10*v[1]/v[2]])
        vertices=result['table_reference']['polygon_base_m']
        self.assertEqual([v[:2] for v in vertices],workspace_polygon(p))
        self.assertTrue(all(v[0]>=.5 and v[1]<=.3 for v in vertices))
        self.assertTrue(all(abs(v[2]-table_height(v[0],v[1],p))<1e-12 for v in vertices))
        self.assertIsNone(result['table_reference']['base_z_m'])
        self.assertGreater(max(v[2] for v in vertices)-min(v[2] for v in vertices),.02)
        self.assertEqual(p,before)

    def test_transit_accounts_for_higher_surface_and_held_bottom(self):
        p=profile();p['transit_z_m']=-.15
        st=state(True);st['eef_m'][2]=-.08
        stages=plan('place',st,p,[.9,.2,-.14])
        travel=[s for s in stages if s['kind']=='move' and s['tcp_m'][0]>.8]
        clearance=table_height(.9,.2,p)+p['table_clearance_m']+p['approach_m']+p['held_bottom_below_tcp_m']
        self.assertGreaterEqual(travel[0]['tcp_m'][2],clearance-1e-12)


class LongPathPreflight(unittest.TestCase):
    def test_long_plan_splits_without_dropping_grasp_or_release_stages(self):
        p=profile();st=state();stages=plan('pick',st,p,[.9,.3,-.12])
        before=st['eef_m'];moves=[]
        for stage in stages:
            if stage['kind']!='move':continue
            dest=eef_from_tcp(stage['tcp_m'],stage['xyzw'],p)
            self.assertLessEqual(math.dist(before,dest),p['max_leg_m']+1e-9)
            before=dest;moves.append(stage)
        self.assertGreater(len(moves),8)
        self.assertEqual([s['open'] for s in stages if s['kind']=='gripper'],[True,False])
        self.assertEqual(stages[-1],{'kind':'verify_grasp','held':True})

    def test_split_stages_consume_budget_before_any_ik_or_motor(self):
        s,r,modules=service();stages=plan('move_to',state(),s.profile,[.9,.3,-.1])
        operations=sum(v['kind'] in ('move','gripper') for v in stages)
        r.motion_remaining=operations-1
        with patch.dict('sys.modules',modules):result=s.prepare('o','move_to',target_base_m=[.9,.3,-.1])
        self.assertEqual(result['failure_code'],'stage_budget');self.assertEqual(result['required_stages'],operations)
        r.continuous_ik.solve.assert_not_called();r.move_segment.assert_not_called()

    def test_late_ik_failure_never_executes_earlier_segments(self):
        s,r,modules=service()
        def solve(point,*_):
            if point.pose.position.x>.85:raise IKUnreachableError('far endpoint unreachable')
            return {'j':0.}
        r.continuous_ik.solve.side_effect=solve
        with patch.dict('sys.modules',modules):result=s.prepare('o','move_to',target_base_m=[.9,.3,-.1])
        self.assertEqual(result['failure_code'],'ik_preflight');self.assertGreater(r.continuous_ik.solve.call_count,30)
        self.assertIsNone(s.prepared);r.move_segment.assert_not_called();r.grip.command_position.assert_not_called()

    def test_peer_hook_runs_after_all_ik_and_before_token(self):
        class PeerRobot(Mock):
            def check_other_arm_path(self,paths):
                self.peer_result=(len(paths),self.continuous_ik.solve.call_count)
                raise RuntimeError('peer overlap')
        s,r,modules=service(robot_type=PeerRobot)
        with patch.dict('sys.modules',modules),self.assertRaisesRegex(RuntimeError,'peer overlap'):
            s.prepare('o','move_to',target_base_m=[.9,.3,-.1])
        self.assertGreater(r.peer_result[0],6);self.assertGreater(r.peer_result[1],50)
        self.assertIsNone(s.prepared);r.move_segment.assert_not_called();r.grip.command_position.assert_not_called()

    def test_rotating_offset_cannot_leave_polygon_between_valid_endpoints(self):
        p=profile();p['tcp_offset_tool_m']=[.1,0,0]
        p['table_polygon_base_xy']=[[.3,.1],[1.,.1],[1.,.25],[.3,.25]]
        s,r,modules=service(p)
        r.limb.endpoint_pose.return_value={'position':NS(x=.6,y=.2,z=.1),'orientation':NS(x=1.,y=0.,z=0.,w=0.)}
        stages=[{'kind':'move','tcp_m':[.7,.2,.1],'xyzw':[0,1,0,0]}]
        p['max_leg_m']=.4
        with patch.dict('sys.modules',modules),self.assertRaisesRegex(ValueError,'Interpolated TCP leaves workspace'):
            s._preflight(stages,{'state':state()},time.monotonic())
        r.move_segment.assert_not_called()

    def test_rotating_offset_cannot_dip_through_plane_between_endpoints(self):
        p=profile();p['tcp_offset_tool_m']=[0,0,.1];p['max_leg_m']=.4
        p['table_plane']={'normal':[0,0,1],'offset':.03}
        p['transit_z_m']=.05
        s,r,modules=service(p)
        a=math.radians(135)/2;b=math.radians(225)/2
        r.limb.endpoint_pose.return_value={'position':NS(x=.6-math.sqrt(.005),y=.2,z=math.sqrt(.005)),
                                          'orientation':NS(x=0.,y=math.sin(a),z=0.,w=math.cos(a))}
        stages=[{'kind':'move','tcp_m':[.6,.2,0.],'xyzw':[0,math.sin(b),0,math.cos(b)]}]
        with patch.dict('sys.modules',modules),self.assertRaisesRegex(ValueError,'below table'):
            s._preflight(stages,{'state':state()},time.monotonic())
        r.move_segment.assert_not_called()


class CandidateProfiles(unittest.TestCase):
    def test_generator_keeps_independent_arms_uncommissioned(self):
        template=json.loads(Path('config/baxter-primitives.example.json').read_text())
        template['primitives']['tool_orientation_xyzw']=[math.sqrt(.5),math.sqrt(.5),0,0]
        geometry={'plane':{'normal':[-.1,0,1],'offset':.25},
                  'interpreted_boundary_base_m':[[.35,-.6,-.215],[1,-.6,-.15],[1,.6,-.15],[.35,.6,-.215]],
                  'camera_to_base_xyzw':[0,0,0,1],'camera_to_base_translation_m':[0,0,1]}
        original=copy.deepcopy(template);out=build_candidate_profiles(template,geometry)
        self.assertEqual(template,original)
        for arm,cfg in out.items():
            self.assertEqual(cfg['arm'],arm);self.assertFalse(cfg['commissioned']);self.assertFalse(cfg['primitives']['calibrated'])
            self.assertFalse(cfg['site_candidate']['tcp_verified']);validate_profile(cfg['primitives'])
            self.assertIsNone(cfg['primitives']['home_tcp_m']);self.assertEqual(cfg['primitives']['observation_tcp_m'],[])
            self.assertGreater(cfg['primitives']['tcp_workspace_max_m'][1]-cfg['primitives']['tcp_workspace_min_m'][1],1.)
        self.assertNotEqual(out['left']['primitives']['tool_orientation_xyzw'],out['right']['primitives']['tool_orientation_xyzw'])
        self.assertEqual(out['right']['primitives']['tcp_offset_tool_m'],[0,0,.0155])
        out['left']['primitives']['table_plane']['offset']=2
        self.assertEqual(out['right']['primitives']['table_plane']['offset'],.25)

    def test_full_xyz_fit_applied_with_provenance_but_never_autoenabled(self):
        template=json.loads(Path('config/baxter-primitives.example.json').read_text())
        geometry={'plane':{'normal':[0,0,1],'offset':.18},
                  'interpreted_boundary_base_m':[[.35,-.6,-.18],[1,-.6,-.18],[1,.6,-.18],[.35,.6,-.18]],
                  'camera_to_base_xyzw':[0,0,0,1],'camera_to_base_translation_m':[0,0,1]}
        fit={'arm':'right','method':'fixed_tip_pivot','tcp_offset_tool_m':[.02,-.01,.12],
             'samples':6,'rms_m':.002,'max_error_m':.004,'commissioned':False}
        result=build_candidate_profiles(template,geometry,{'right':fit})['right']
        self.assertEqual(result['primitives']['tcp_offset_tool_m'],fit['tcp_offset_tool_m'])
        self.assertEqual(result['site_candidate']['tcp_fit'],fit)
        self.assertFalse(result['commissioned']);self.assertFalse(result['primitives']['calibrated'])
        self.assertFalse(result['site_candidate']['tcp_verified'])
        for bad in ({**fit,'arm':'left'},{**fit,'rms_m':.02},{**fit,'samples':3}):
            with self.assertRaises(ValueError):build_candidate_profiles(template,geometry,{'right':bad})


if __name__=='__main__':unittest.main()
