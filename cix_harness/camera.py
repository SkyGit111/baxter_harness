# Copyright (c) 2026 Baxter-Claw Contributors. MIT; see third_party/Baxter-claw-LICENSE.
# Adapted from bridge/drivers/realsense_driver.py for fresh RGB-only policy observations.
import time

from .contract import image_data


class D455Camera:
    def __init__(self, serial=""):
        import pyrealsense2 as rs

        self.pipeline = rs.pipeline()
        config = rs.config()
        if serial:
            config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile=self.pipeline.start(config)
        self.align=rs.align(rs.stream.color)
        self.depth_scale=profile.get_device().first_depth_sensor().get_depth_scale()
        self.intrinsics=profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.last_depth=None
        self.last_number = -1
        # Do not advertise a ready bridge before the device actually streams.
        try:
            frames=self.pipeline.wait_for_frames(5000)
            if not frames.get_color_frame() or not frames.get_depth_frame():
                raise RuntimeError('D455 startup requires color and depth frames')
        except BaseException:
            self.pipeline.stop()
            raise

    def capture(self):
        import cv2
        import numpy as np

        # Discard the SDK's newest buffered frame, which may predate the last
        # motion. The subsequent frame must have a greater device frame number.
        old = self.pipeline.wait_for_frames(1500).get_color_frame()
        if not old:
            raise RuntimeError("D455 returned no color frame")
        number = old.get_frame_number()
        captured = time.monotonic()
        frames=self.align.process(self.pipeline.wait_for_frames(1000))
        frame = frames.get_color_frame()
        if not frame or frame.get_frame_number() <= max(number, self.last_number):
            raise RuntimeError("Repeated or missing D455 color frame")
        self.last_number = frame.get_frame_number()
        depth=frames.get_depth_frame()
        self.last_depth=None
        if depth:
            self.last_depth=np.asanyarray(depth.get_data()).copy()*self.depth_scale
            self.last_color=np.asanyarray(frame.get_data()).copy()
        ok, jpeg = cv2.imencode(".jpg", np.asanyarray(frame.get_data()), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok or time.monotonic() - captured >= 0.5:
            raise RuntimeError("Could not obtain a fresh D455 JPEG")
        return image_data(jpeg.tobytes()), captured, self.last_number

    def save_depth(self, path):
        import numpy as np
        if self.last_depth is None:
            raise RuntimeError('No aligned depth for current camera frame')
        i=self.intrinsics
        np.savez_compressed(path,depth_m=self.last_depth,bgr=self.last_color,
                            intrinsics=np.array([i.fx,i.fy,i.ppx,i.ppy]),frame_number=self.last_number)

    def close(self):
        self.pipeline.stop()
