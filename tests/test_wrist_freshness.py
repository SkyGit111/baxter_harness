import threading
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

from cix_harness.baxter import Baxter


class Stamp(float):
    def __sub__(self, other):
        return Stamp(float(self)-other)

    def to_sec(self):
        return float(self)


class WristFreshnessTest(unittest.TestCase):
    def setUp(self):
        self.arm=object.__new__(Baxter)
        self.arm.cfg={'commissioning_scope':'supervised_primitive_task','arm':'left','robot_clock_offset_s':.038}
        self.arm.ros=Mock()
        self.arm.ros.Time.now.return_value=Stamp(100)
        self.arm.state_stamps={}
        self.arm.wrist_state=None
        self.arm.wrist_condition=threading.Condition()
        self.arm.camera=Mock()
        self.arm.camera.capture.return_value=('scene',200.,8)
        self.arm.read_state=lambda number: {'joint_names':['j'],'joint_rad':[0.]}
        self.message=SimpleNamespace(header=SimpleNamespace(seq=15,stamp=Stamp(99.502)))
        self.clock=patch('cix_harness.baxter.time.monotonic',return_value=200.)
        self.clock.start()
        self.addCleanup(self.clock.stop)

    def test_auxiliary_delivery_delay_is_not_a_new_stamp_or_gripper_age(self):
        self.arm.note_wrist(self.message)
        self.assertAlmostEqual(self.arm.wrist_state[0],199.464)
        self.assertIsNone(self.arm.telemetry_time(self.message.header.stamp,'joints'))
        self.assertIsNone(self.arm.telemetry_time(self.message.header.stamp,'gripper'))
        cv=Mock()
        encoded=Mock()
        encoded.tobytes.return_value=bytes([255,216,255])
        cv.imencode.return_value=(True,encoded)
        with patch.dict('sys.modules',cv2=cv,cv_bridge=Mock()):
            obs=self.arm.observe()
        self.assertEqual(obs['camera_metadata']['wrist']['frame_number'],15)
        self.assertAlmostEqual(obs['state']['wrist_age_s'],.536)
        self.arm.wrist_condition=MagicMock()
        repeat=self.arm.observe()
        self.arm.wrist_condition.wait_for.assert_called_once()
        self.assertNotIn('wrist_image',repeat)
        self.assertEqual(repeat['camera_metadata']['wrist']['omitted_reason'],'already_used')

    def test_wait_for_post_motion_frame_before_capturing_scene(self):
        self.arm.note_wrist(self.message)
        self.arm.wrist_after_motion=199.5
        self.arm.wrist_condition=MagicMock()
        events=[]
        def arrival(predicate,timeout):
            self.assertLessEqual(timeout,.6)
            self.assertFalse(predicate())
            events.append('new_wrist')
            msg=SimpleNamespace(header=SimpleNamespace(seq=16,stamp=Stamp(99.65)))
            self.arm.note_wrist(msg)
            self.assertTrue(predicate())
        self.arm.wrist_condition.wait_for.side_effect=arrival
        self.arm.camera.capture.side_effect=lambda: (events.append('scene') or ('scene',200.,9))
        cv=Mock()
        encoded=Mock()
        encoded.tobytes.return_value=bytes([255,216,255])
        cv.imencode.return_value=(True,encoded)
        with patch.dict('sys.modules',cv2=cv,cv_bridge=Mock()):
            obs=self.arm.observe()
        self.assertEqual(events,['new_wrist','scene'])
        self.assertEqual(obs['camera_metadata']['wrist']['frame_number'],16)

    def test_required_policy_wrist_keeps_half_second_cutoff(self):
        self.arm.cfg.update(commissioning_scope='supervised_hybrid_task',require_wrist=True)
        self.arm.note_wrist(self.message)
        self.assertIsNone(self.arm.wrist_state)
        with self.assertRaisesRegex(RuntimeError,'required by Pi05'):
            self.arm.observe()
        self.arm.cfg['require_wrist']=False
        self.arm.note_wrist(self.message)
        self.assertIsNotNone(self.arm.wrist_state)
        self.assertNotIn('wrist_image',self.arm.observe())
