import itertools
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch
import xml.etree.ElementTree as ET

import numpy as np

from cix_harness import handeye_drive as drive


class CalibrationDriveTest(unittest.TestCase):
    def test_waits_for_new_stable_endpoint_messages_after_joint_settle(self):
        driver=object.__new__(drive.Driver)
        driver.check_ownership=Mock()
        now=[100.]
        pose=lambda x:{'position':NS(x=x,y=0.,z=0.),'orientation':NS(x=0.,y=0.,z=0.,w=1.)}
        driver.states={'endpoint':(99.9,pose(.008))}
        def receive():
            elapsed=now[0]-100
            if elapsed>.035:
                driver.states['endpoint']=(now[0],pose(.006 if elapsed<.08 else .001))
        driver.live=Mock(side_effect=receive)
        def elapse(s):now[0]+=s
        with patch.dict('sys.modules',{'baxter_core_msgs.msg':NS(JointCommand=object)}), \
                patch.object(drive.time,'monotonic',side_effect=lambda:now[0]), \
                patch.object(drive.time,'sleep',side_effect=elapse):
            driver.execute([],{},np.eye(4))
            self.assertGreaterEqual(now[0],100.16)
            # A repeated once-valid measurement cannot certify a fresh endpoint.
            driver.live=Mock()
            with self.assertRaisesRegex(RuntimeError,'Fresh stable endpoint'):
                driver.execute([],{},np.eye(4))
        self.assertGreater(driver.check_ownership.call_count,2)

    def test_joint_route_can_use_fully_checked_anchor_when_direct_chord_fails(self):
        obj=object.__new__(drive.Driver);obj.anchor_joints={'x':0.,'y':0.}
        obj.record=Mock()
        def check(q):
            if q['x']>.003 and q['y']>.003:raise drive.CandidateRejected('Blocked diagonal')
        obj.check=Mock(side_effect=check)
        path=obj.joint_route({'x':.02,'y':0.},{'x':0.,'y':.02})
        self.assertIn(obj.anchor_joints,path)
        for p in path:check(p)
        obj.record.assert_called_once()

    def test_wrist_lattice_bounded_shift_and_separated_subset(self):
        anchor=np.eye(4)
        poses=list(drive.wrist_lattice(anchor))
        self.assertGreater(len(poses),100)
        for p in poses:
            self.assertLessEqual(np.linalg.norm(p[:3,3]),.01600001)
            self.assertGreaterEqual(p[2,3],-.003001)
        # Representative wrist cone from an extended forearm.
        feasible=[(p,{}) for p in poses if drive.angle(p[:3,:3])<math.radians(24)
                  and p[2,2]>math.cos(math.radians(12))]
        selected=drive.separated_joint_goals(feasible,8)
        self.assertEqual(len(selected),8)
        for (p,_),(q,_) in itertools.combinations(selected,2):
            self.assertGreaterEqual(drive.angle(p[:3,:3].T@q[:3,:3]),math.radians(15)-1e-8)

    def test_wrist_lattice_preview_dispatches_no_motor_commands(self):
        with TemporaryDirectory() as folder:
            robot=Mock();robot.anchor=np.eye(4);robot.anchor_joints={'j':0}
            robot.lattice_route.return_value=[(np.eye(4),{'j':.1})]
            with patch.object(drive,'Driver',return_value=robot):
                self.assertEqual(drive.main(['--wrist-lattice','--count','1','--tool-radius-m','.14',
                                             '--clearance-m','.095','--body-clearance-m','.09',
                                             '--output',str(Path(folder)/'preview.jsonl')]),0)
            robot.lattice_route.assert_called_once();robot.enable_publisher.assert_not_called()

    def test_anchor_uses_an_actual_admissible_sample_without_clipping(self):
        obj=object.__new__(drive.Driver)
        obj.ik=Mock(names=['right_w1'],lower=[-1.57],upper=[2.09])
        legal={'right_w1':-1.5694}
        obj.live=Mock(side_effect=[{'right_w1':-1.5712},legal])
        self.assertIs(obj.in_limit_sample(drive.time.monotonic()+1),legal)
        with self.assertRaisesRegex(RuntimeError,'URDF limits'):
            obj.in_limit_sample(drive.time.monotonic())

    def test_recenter_requires_legal_edge_and_only_increases_w1(self):
        obj=object.__new__(drive.Driver)
        obj.ik=Mock(names=['right_w1','right_e0'],lower=[-1.57,-2],upper=[2,2])
        obj.ik.forward.return_value={'position':NS(x=.5,y=.1,z=.3),'orientation':NS(x=0,y=0,z=0,w=1)}
        def reject_far(q):
            if q['right_w1']> -1.47:raise drive.CandidateRejected('sweep exceeds local workspace')
        obj.check=Mock(side_effect=reject_far)
        start={'right_w1':-1.5695,'right_e0':-.8}
        path,_=obj.recenter(start)
        self.assertGreater(path[-1]['right_w1'],start['right_w1']+.035)
        self.assertTrue(all(q['right_e0']==start['right_e0'] for q in path))
        self.assertTrue(all(b['right_w1']>a['right_w1'] for a,b in zip([start]+path,path)))
        for inadmissible in (-1.571,-1.1):
            with self.assertRaisesRegex(ValueError,'observed legal'):
                obj.recenter(dict(start,right_w1=inadmissible))

    def test_planned_views_are_distinct_capture_poses_about_the_anchor(self):
        anchor=np.eye(4);anchor[:3,3]=[.45,.19,.2]
        poses=drive.targets(anchor,19)
        for a,b in itertools.combinations(poses,2):
            self.assertGreater(drive.angle(a[:3,:3].T@b[:3,:3]),math.radians(12))
        for p in poses:self.assertLess(drive.check_sweep(anchor,p,.2,.13),.1)
        with self.assertRaises(drive.CandidateRejected):drive.check_sweep(anchor,poses[1],.2,.05)

    def test_inward_command_uses_real_in_limit_sample_and_rejects_other_excursions(self):
        obj = object.__new__(drive.Driver)
        obj.ik = NS(names=['right_w1','right_e0'], lower=[-1.57,-2], upper=[2,2])
        bad = {'right_w1':-1.571, 'right_e0':0.}
        good = dict(bad, right_w1=-1.569)
        obj.live = Mock(side_effect=[bad,good])
        obj.check_ownership = Mock()
        self.assertIs(obj.command_sample(True), good)
        obj.check_ownership.assert_called_once()
        for inward, state in [(False,bad),(True,dict(bad,right_w1=-1.58)),(True,dict(good,right_e0=2.001))]:
            obj.live = Mock(return_value=state)
            with self.assertRaisesRegex(RuntimeError,'outside URDF'):
                obj.command_sample(inward)

    def test_motion_knots_and_attachment_cover_bounds(self):
        values=list(drive.joint_path({'one':.1},{'one':.13}))
        xs=[.1]+[v['one'] for v in values]
        self.assertAlmostEqual(xs[-1],.13)
        self.assertLessEqual(max(b-a for a,b in zip(xs,xs[1:])),.0050001)
        xml='<robot><link name="joint"><collision><origin xyz="0 .02 0"/><geometry><box size=".06 .08 .01"/></geometry></collision></link></robot>'
        shape=drive.link_shapes(xml,{'joint':1})[0][1][0]
        moved=np.eye(4);moved[0,3]=.08
        self.assertAlmostEqual(drive.shape_sweep(np.eye(4),moved,shape),.08)
        with TemporaryDirectory() as folder:
            file=Path(folder)/'boxes.json'
            file.write_text(json.dumps({'frame':'right_gripper','units':'m','boxes':[
                {'size_m':[.12,.2,.005],'xyz':[0,0,.03],'xyzw':[0,0,0,1]}]}))
            self.assertEqual(len(drive.attachment_boxes(file,.2)),1)
            with self.assertRaises(ValueError):drive.attachment_boxes(file,.09)

    def test_rotated_offset_cylinder_and_fixed_side_branch(self):
        xml='''<robot><link name="wrist"/><link name="side"><collision>
        <origin rpy="0 1.5707963267948966 0" xyz=".1 0 0"/>
        <geometry><cylinder length=".2" radius=".06"/></geometry></collision></link>
        <joint name="side_joint" type="fixed"><parent link="wrist"/><child link="side"/>
        <origin xyz="0 .03 0"/></joint></robot>'''
        shape=drive.link_shapes(xml,{'wrist':1})[0][1][0]
        self.assertEqual(shape[0],'side')
        spin=drive.urdf_pose(ET.Element('origin',rpy='1 0 0'))
        # Translate so the off-centre cylinder's axis is the rotation axis.
        centre=shape[1][:3,3]
        spin[:3,3]=centre-spin[:3,:3]@centre
        self.assertAlmostEqual(drive.shape_sweep(np.eye(4),spin,shape),0.)
        tilt=drive.urdf_pose(ET.Element('origin',rpy='0 .3 0'))
        self.assertGreater(drive.shape_sweep(np.eye(4),tilt,shape),.05)

    def test_preview_skips_rejected_candidates_without_publishing(self):
        with TemporaryDirectory() as folder:
            log=Path(folder)/'new.jsonl'
            robot=Mock();robot.anchor=np.eye(4);robot.anchor_joints={'j':.1}
            robot.plan.side_effect=[drive.CandidateRejected('collision'),[{'j':.1}],[{'j':.1}]]
            with patch.object(drive,'Driver',return_value=robot):
                self.assertEqual(drive.main(['--count','2','--tool-radius-m','.2','--clearance-m','.3',
                                             '--body-clearance-m','.12','--output',str(log)]),0)
            events=[json.loads(line)['event'] for line in log.read_text().splitlines()]
            self.assertEqual(events,['anchor','candidate_rejected','preflight','preflight'])
            robot.enable_publisher.assert_not_called();robot.execute.assert_not_called()

    def test_stale_feedback_aborts_all_candidates(self):
        with TemporaryDirectory() as folder:
            robot=Mock();robot.anchor=np.eye(4);robot.anchor_joints={'j':0}
            robot.plan.side_effect=RuntimeError('Stale telemetry')
            with patch.object(drive,'Driver',return_value=robot),self.assertRaisesRegex(RuntimeError,'Stale'):
                drive.main(['--tool-radius-m','.2','--clearance-m','.3','--body-clearance-m','.1',
                            '--output',str(Path(folder)/'new.jsonl')])
            robot.enable_publisher.assert_not_called()
