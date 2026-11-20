import io
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from cix_harness.camera import D455Camera


class CameraSnapshotTest(unittest.TestCase):
    def test_snapshot_preserves_calibration_and_original_depth_timing(self):
        camera = object.__new__(D455Camera)
        camera.serial = "sensor-001"
        camera.depth_scale = 0.001
        camera.last_number = 7
        camera.pipeline = Mock()
        camera.align = Mock()
        old, raw, aligned = Mock(), Mock(), Mock()
        camera.pipeline.wait_for_frames.side_effect = [old, raw]
        camera.align.process.return_value = aligned
        old.get_color_frame.return_value.get_frame_number.return_value = 8
        color = aligned.get_color_frame.return_value
        color.get_frame_number.return_value = 9
        color.get_timestamp.return_value = 550.
        color.get_frame_timestamp_domain.return_value = 'hardware_clock'
        bgr = np.zeros((2, 3, 3), dtype=np.uint8)
        color.get_data.return_value = bgr
        color.profile.as_video_stream_profile.return_value.get_intrinsics.return_value = SimpleNamespace(
            fx=4.2, fy=4.1, ppx=1.6, ppy=1.1, width=3, height=2,
            model='distortion.inverse_brown_conrady', coeffs=[-.05,.07,0.,0.,-.02])
        depth = raw.get_depth_frame.return_value
        depth.get_frame_number.return_value = 22
        depth.get_timestamp.return_value = 549.
        depth.get_frame_timestamp_domain.return_value = 'hardware_clock'
        aligned.get_depth_frame.return_value.get_data.return_value = np.full((2,3), 800, dtype=np.uint16)
        cv = Mock()
        cv.imencode.return_value = (True, np.array([255, 216, 255], dtype=np.uint8))
        archive = io.BytesIO()
        with patch.dict('sys.modules', cv2=cv):
            _, _, number = camera.capture()
        camera.save_depth(archive)
        archive.seek(0)
        with np.load(archive, allow_pickle=False) as snap:
            self.assertEqual(number, snap['frame_number'])
            self.assertEqual(int(snap['depth_frame_number']), 22)
            self.assertEqual(float(snap['color_timestamp_ms'] - snap['depth_timestamp_ms']), 1.)
            self.assertEqual(str(snap['color_timestamp_domain']), str(snap['depth_timestamp_domain']))
            self.assertEqual(str(snap['camera_serial']), camera.serial)
            self.assertEqual(str(snap['distortion_model']), 'distortion.inverse_brown_conrady')
            np.testing.assert_allclose(snap['intrinsics'], [4.2, 4.1, 1.6, 1.1])
            np.testing.assert_allclose(snap['K'], [[4.2,0,1.6],[0,4.1,1.1],[0,0,1]])
            np.testing.assert_allclose(snap['D'], [-.05,.07,0,0,-.02])
            np.testing.assert_allclose(snap['depth_m'], .8)
