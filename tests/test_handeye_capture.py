import math
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from collections import deque
from types import SimpleNamespace as NS
import threading
from unittest.mock import Mock, patch

import numpy as np

from cix_harness.handeye_capture import EndpointReader, capture, gate_pose, pose_difference, refine_tag_edges, stationary, tag_from_depth
from cix_harness.handeye_capture import load_volume_checkpoint, volume_capture_plan


class HandeyeCapture(unittest.TestCase):
    def test_volume_resume_preserves_frozen_groups_camera_and_observation_ids(self):
        with TemporaryDirectory() as folder:
            root=Path(folder)
            pose={'xyz':[0.,0.,0.], 'xyzw':[0.,0.,0.,1.]}
            rows=[{'id':str(i),'cell':[i%3,0,0],'base_T_effector':dict(pose,xyz=[i*.04,0,0]),
                   'split':'holdout' if i%5==0 else 'train'} for i in range(30)]
            plan=root/'plan.json'
            plan.write_text(json.dumps({'poses':rows}))
            poses,digest,counts=volume_capture_plan(plan)
            self.assertEqual(counts,{'train':24,'holdout':6})
            manifest={'volume_plan_sha256':digest,'camera_serial':'abc'}
            (root/'manifest.json').write_text(json.dumps(manifest))
            holdout=[{'drive_pose_id':'0','drive_target_base_T_effector':np.eye(4).tolist(),'frame_id':'pose-001',
                      'volume_plan_sha256':digest,'volume_split':'holdout','volume_cell':[0,0,0]}]
            for suffix in ('.jpg','.npz'):(root/('pose-001'+suffix)).touch()
            (root/'train.json').write_text('[]')
            (root/'holdout.json').write_text(json.dumps(holdout))
            self.assertEqual(load_volume_checkpoint(root,manifest,poses)['holdout'],holdout)
            with self.assertRaisesRegex(ValueError,'changed'):
                load_volume_checkpoint(root,dict(manifest,camera_serial='other'),poses)
            (root/'train.json').write_text(json.dumps(holdout))
            with self.assertRaisesRegex(ValueError,'Inconsistent'):
                load_volume_checkpoint(root,manifest,poses)

    def scene(self, roll=0.):
        fx,fy,px,py=230.,235.,100.,80.
        tilt=.30
        rx=np.array([[1,0,0],[0,math.cos(tilt),-math.sin(tilt)],[0,math.sin(tilt),math.cos(tilt)]])
        rz=np.array([[math.cos(roll),-math.sin(roll),0],[math.sin(roll),math.cos(roll),0],[0,0,1]])
        rotation=rx @ rz
        center=np.array([.02,-.01,.66])
        local=np.array([[-.10,-.08,0],[.10,-.08,0],[.10,.08,0],[-.10,.08,0]])
        xyz=local @ rotation.T+center
        uv=xyz[:,:2]/xyz[:,2:3]*[fx,fy]+[px,py]
        ray=lambda uv:np.array([(uv[0]-px)/fx,(uv[1]-py)/fy,1.])
        v,u=np.indices((160,200))
        normal=rotation[:,2]
        depth=(normal @ center)/((u-px)/fx*normal[0]+(v-py)/fy*normal[1]+normal[2])
        expected=np.eye(4)
        expected[:3,3]=center
        expected[:3,:3]=rotation
        return uv,depth,ray,expected

    def test_plane_with_outliers_recovers_decoded_corner_axes_and_optical_depth(self):
        for angle in (0.,np.pi/2,np.pi):
            corners,depth,ray,expected=self.scene(angle)
            # Contaminate part of the tag with aligned background and missing returns.
            depth[:,83:89]=1.2
            depth[77:81,:]=0
            result=tag_from_depth(corners,depth,ray,.20,.16)
            np.testing.assert_allclose(result['camera_T_tag'],expected,atol=1e-10)
            self.assertAlmostEqual(result['width_m'],.2)
            self.assertAlmostEqual(result['height_m'],.16)
            self.assertLess(result['plane_rms_m'],1e-10)
            self.assertLess(result['plane_inliers'],result['sampled_pixels'])

    def test_unusable_depth_and_ruler_disagreement_reject(self):
        uv,depth,ray,_=self.scene()
        with self.assertRaisesRegex(ValueError,'Insufficient valid'):
            tag_from_depth(uv,depth*0,ray)
        with self.assertRaisesRegex(ValueError,'disagrees with ruler'):
            tag_from_depth(uv,depth,ray,.27,.16)

    def test_edge_intersections_correct_inconsistent_coarse_corners(self):
        gray=np.full((120,160),240,dtype=np.uint8)
        gray[35:85,45:105]=10
        gray[50:65,60:82]=255
        # Preserve the decoded orientation, including when its corner 0 is at bottom right.
        exact=np.array([[104.5,84.5],[44.5,84.5],[44.5,34.5],[104.5,34.5]])
        coarse=exact+np.array([[-1.,2.],[-2.,-1.],[2.,-2.],[1.,1.]])
        refined,stats=refine_tag_edges(gray,coarse)
        self.assertLess(np.max(np.linalg.norm(refined-exact,axis=1)),.8)
        self.assertEqual(len(stats),4)
        with self.assertRaisesRegex(ValueError,'contrast'):
            refine_tag_edges(gray*0,coarse)

    def test_stationary_window_requires_live_multiframe_camera_and_ros_data(self):
        eye=np.eye(4)
        frames=[dict(host_time_s=5+i*.1,endpoint_ros_stamp_s=i/10,joint_ros_stamp_s=i/10,
                     color_stamp_ms=100*i,depth_stamp_ms=100*i,camera_T_tag=eye,base_T_effector=eye)
                for i in range(7)]
        self.assertTrue(stationary(deque(frames)))
        self.assertFalse(stationary(frames[:3]))
        self.assertFalse(stationary([dict(f,joint_ros_stamp_s=10.) for f in frames]))
        rotated=self.scene(.5)[3]
        self.assertFalse(stationary(frames[:-1]+[dict(frames[-1],camera_T_tag=rotated)]))
        self.assertFalse(stationary(frames[:-1]+[dict(frames[-1],base_T_effector=rotated)]))
        self.assertAlmostEqual(pose_difference(rotated,rotated)[0],0.)

    def test_gate_requires_current_target_id_and_measured_pose(self):
        target=np.eye(4)
        with TemporaryDirectory() as folder:
            path=Path(folder)/'gate.json'
            with self.assertRaisesRegex(ValueError,'Waiting for valid'):
                gate_pose(path,target)
            path.write_text(json.dumps({'id':'r1','base_T_effector':{'xyz':[0,0,0],'xyzw':[0,0,0,1]}}))
            self.assertEqual(gate_pose(path,target)[0],'r1')
            target[0,3]=.006
            with self.assertRaisesRegex(ValueError,'Waiting for target'):
                gate_pose(path,target)

    def test_reads_only_complete_and_fresh_joint_feedback_and_keeps_outlier(self):
        r=EndpointReader.__new__(EndpointReader)
        r.lock=threading.Lock()
        q=NS(x=0.,y=0.,z=0.,w=1.);p=NS(x=.1,y=.2,z=.3)
        r.fk=NS(names=['right_w1'],lower=[-1.57],upper=[2.],forward=lambda j:{'position':p,'orientation':q})
        msg=lambda t: NS(stamp=NS(to_sec=lambda:t),frame_id='')
        ep=NS(header=msg(10.),pose=NS(position=p,orientation=q))
        near=NS(header=msg(10.),name=['right_w1'],position=[-1.571])
        good=NS(header=msg(9.99),name=['right_w1'],position=[-1.569])
        r.buffers={'endpoint':deque([(1.,ep)]),'joints':deque([(.99,good),(1.,near)])}
        r.receive(NS(name=['left_gripper'],position=[.2]),'joints')
        self.assertEqual(len(r.buffers['joints']),2)
        result=r.sample(1.)
        self.assertEqual(result['pose_check'],'urdf_fk')
        self.assertEqual(result['skipped_joint_samples'][0]['outside_rad'],{'right_w1':-1.571})
        self.assertEqual(result['joint_ros_stamp_s'],9.99)
        r.buffers['joints']=deque([(1.,near)])
        r.tf=Mock();r.tf_errors=(LookupError,);r.rospy=NS(Time=lambda x:x)
        tf_msg=NS(header=msg(10.),transform=NS(translation=p,rotation=q));r.tf.lookup_transform.return_value=tf_msg
        result=r.sample(1.)
        self.assertEqual(result['pose_check'],'tf_readonly')
        self.assertIsNone(result['fk_error_m'])
        self.assertEqual(result['joint_limit_violations']['right_w1']['lower_rad'],-1.57)
        tf_msg.header=msg(9.)
        with self.assertRaisesRegex(ValueError,'no fresh independent TF'):
            r.sample(1.)

    def test_capture_writes_measured_sample_and_closes_pipeline(self):
        # Exercise the actual window, gating and persistence loop with hardware interfaces stubbed.
        clock=[100.]
        def advance(_timeout):
            clock[0]+=.11
            return frames
        intr=NS(width=160,height=120,ppx=80.,ppy=60.,fx=500.,fy=500.,coeffs=[0.]*5,model='pinhole')
        gray=np.full((120,160),250,dtype=np.uint8); gray[35:85,50:100]=5
        corner=np.array([[[50,35],[99,35],[99,84],[50,84]]],np.float32)
        cv=Mock()
        cv.aruco.ArucoDetector.return_value.detectMarkers.return_value=([corner],np.array([[0]]),[])
        cv.cvtColor.return_value=gray
        cv.imwrite.side_effect=lambda path,frame:Path(path).write_bytes(b'fixture')
        rs=Mock()
        pipe=rs.pipeline.return_value
        pipe.wait_for_frames.side_effect=advance
        source=pipe.start.return_value
        source.get_device.return_value.first_depth_sensor.return_value.get_depth_scale.return_value=.001
        profile=source.get_stream.return_value
        profile.as_video_stream_profile.return_value.get_intrinsics.return_value=intr
        profile.get_extrinsics_to.return_value=NS(rotation=np.eye(3).ravel().tolist(),translation=[0.,0.,0.])
        frames=Mock();rs.align.return_value.process.side_effect=lambda f:f
        color,depth=frames.get_color_frame.return_value,frames.get_depth_frame.return_value
        depth.profile=profile
        for frame in (color,depth):
            frame.get_frame_number.side_effect=lambda:int(10*clock[0])
            frame.get_timestamp.side_effect=lambda:clock[0]*1000
            frame.get_frame_timestamp_domain.return_value='device'
        color.get_data.return_value=np.repeat(gray[:,:,None],3,axis=2)
        depth.get_data.return_value=np.full(gray.shape,1000,np.uint16)
        rs.rs2_deproject_pixel_to_point.side_effect=lambda i,p,z:[(p[0]-i.ppx)/i.fx,(p[1]-i.ppy)/i.fy,z]
        pose=np.eye(4).tolist()
        robot=Mock(subs=[])
        robot.sample.side_effect=lambda arrival:{'base_T_effector':pose,'endpoint_ros_stamp_s':arrival,
                'joint_ros_stamp_s':arrival,'fk_error_m':0.,'pose_check':'urdf_fk','joint_limit_violations':{}}
        ros=Mock();ros.is_shutdown.return_value=False
        with TemporaryDirectory() as folder:
            root=Path(folder); out=root/'new'; gate=root/'gate.json'
            gate.write_text(json.dumps({'id':'sample-1','base_T_effector':{'xyz':[0,0,0],'xyzw':[0,0,0,1]}}))
            args=NS(output_dir=out,serial='fake',train=1,holdout=0,duration_s=5.,color_mode='160x120',
                    depth_mode='160x120',pose_gate=gate,width_m=.10,height_m=.10)
            with patch.dict('sys.modules',cv2=cv,pyrealsense2=rs,rospy=ros),patch(
                    'cix_harness.handeye_capture.EndpointReader',return_value=robot),patch(
                    'cix_harness.handeye_capture.time.monotonic',side_effect=lambda:clock[0]),patch('builtins.print'):
                self.assertEqual(capture(args),0)
            record=json.loads((out/'train.json').read_text())[0]
            self.assertEqual(record['drive_pose_id'],'sample-1')
            self.assertGreaterEqual(len(record['stationary_frames']),4)
            self.assertEqual(json.loads((out/'holdout.json').read_text()),[])
            self.assertTrue((out/'pose-001.npz').exists())
            self.assertIn('aligned_depth_intrinsics',json.loads((out/'manifest.json').read_text()))
            pipe.stop.assert_called_once()
