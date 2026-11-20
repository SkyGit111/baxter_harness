import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock, patch

import numpy as np

from cix_harness.handeye import packed
from cix_harness.handeye_spatial_drive import (
    CandidateRejected, SpatialDriver, bounds_of_shape, covered, existing_samples,
    marker_projection, validate_scene,
)
from cix_harness.moveit_probe import joint_names


class SpatialDriveTests(unittest.TestCase):
    def setUp(self):
        self.anchor = np.eye(4)
        self.anchor[2,3] = 1.
        self.scene = dict(frame='base', units='m', source='fixture', tool_radius_m=.14,
                          world_boxes=[{'id':'table','lower':[0,0,-.2],'upper':[.8,.8,-.1]}],
                          verified_free_sweep_boxes=[], anchor=dict(base_T_effector=packed(self.anchor),
                            body_clearance_m=.08,tool_clearance_m=.08), base_T_camera=packed(np.eye(4)),
                          effector_T_tag=packed(np.eye(4)),
                          intrinsics=dict(width=640,height=480,fx=400,fy=400,cx=320,cy=240))

    def test_visible_projection_and_actual_capture_are_distinct(self):
        validate_scene(self.scene)
        p = marker_projection(self.anchor,self.scene)
        self.assertEqual(p['corners_px'][0],[300.,220.])
        self.assertIs(p['occlusion_verified'],False)
        later=self.anchor.copy();later[0,3]=1.
        with self.assertRaises(CandidateRejected):
            marker_projection(later,self.scene)
        flipped=self.anchor.copy();flipped[:3,:3]=np.diag([1.,-1.,-1.])
        with self.assertRaises(CandidateRejected):
            marker_projection(flipped,self.scene)

    def test_native_urdf_cylinder_support_in_rotated_frame(self):
        frame=np.eye(4);frame[:3,:3]=[[0,0,1],[1,0,0],[0,1,0]]
        lo,hi=bounds_of_shape(frame,('wrist',np.eye(4),'cylinder',[.05,.20]))
        np.testing.assert_allclose(lo,[-.20,-.05,-.05])
        np.testing.assert_allclose(hi,[.20,.05,.05])
        self.assertFalse(covered(lo,hi,[(lo+.001,hi+.001)]))
        self.assertTrue(covered(lo,hi,[(lo,hi)]))

    def test_cannot_skip_wrong_split_or_missing_capture_evidence(self):
        row={'id':'v01','split':'train','base_T_effector':packed(self.anchor)}
        record={'drive_pose_id':'v01','drive_target_base_T_effector':self.anchor.tolist(),'frame_id':'pose-001'}
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp)
            (root/'holdout.json').write_text('[]')
            (root/'train.json').write_text(json.dumps([record]))
            with self.assertRaisesRegex(ValueError,'evidence'):
                existing_samples(root,{'poses':[row]})
            for ext in ('.npz','.jpg'):
                (root/('pose-001'+ext)).touch()
            self.assertEqual(existing_samples(root,{'poses':[row]}),{'v01'})
            row['split']='holdout'
            with self.assertRaisesRegex(ValueError,'incorrectly split'):
                existing_samples(root,{'poses':[row]})

    def test_plan_only_goal_and_densified_moveit_path(self):
        d=object.__new__(SpatialDriver)
        d.world=['survey geometry'];d.gaps=set();d.gap_bounds={};d.survey={}
        names=sorted(joint_names('right'))
        start=dict.fromkeys(names,0.)
        d.scene=NS(joint_state=NS(name=names+['head_pan'],position=[0.]*8),is_diff=False,
                   attached_collision_objects=['tool'])
        d.live=lambda:dict(start,head_pan=.42)
        d.check=Mock()
        d.ik=NS(forward=lambda q: {'position':NS(x=0.,y=0.,z=1.),
                                  'orientation':NS(x=0.,y=0.,z=0.,w=1.)})
        d.ros=NS(Duration=float);d.move_group=Mock()
        trajectory=NS(joint_names=names,points=[NS(positions=[0.]*7),NS(positions=[.03]*7)])
        d.move_group.get_result.return_value=NS(error_code=NS(val=1),planned_trajectory=NS(joint_trajectory=trajectory))
        def goal():
            return NS(request=NS(),planning_options=NS(plan_only=False,planning_scene_diff=NS(
                      robot_state=NS(),world=NS())))
        genpy=NS(message=NS(fill_message_args=lambda req, data:None))
        with patch.dict('sys.modules',{'genpy':genpy,'genpy.message':genpy.message,
                                     'moveit_msgs.msg':NS(MoveGroupGoal=goal)}):
            path=d.moveit_path(start,self.anchor)
        sent=d.move_group.send_goal.call_args.args[0]
        self.assertIs(sent.planning_options.plan_only,True)
        self.assertEqual(sent.planning_options.planning_scene_diff.world.collision_objects,['survey geometry'])
        self.assertEqual(sent.request.start_state.joint_state.position[-1],.42)
        self.assertEqual(sent.request.start_state.attached_collision_objects,['tool'])
        self.assertGreater(len(path),5)
        self.assertEqual(d.check.call_count,len(path)+1)
