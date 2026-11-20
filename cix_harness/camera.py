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
        self.serial=profile.get_device().get_info(rs.camera_info.serial_number)
        self.intrinsics=profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.last_depth=None
        self.ring_depth_window=None
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
        frame,captured=self._aligned_pair()
        ok, jpeg = cv2.imencode(".jpg", np.asanyarray(frame.get_data()), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok or time.monotonic() - captured >= 0.5:
            raise RuntimeError("Could not obtain a fresh D455 JPEG")
        return image_data(jpeg.tobytes()), captured, self.last_number

    def _aligned_pair(self):
        import numpy as np
        self.ring_depth_window=None

        # Discard the SDK's newest buffered frame, which may predate the last
        # motion. The subsequent frame must have a greater device frame number.
        old = self.pipeline.wait_for_frames(1500).get_color_frame()
        if not old:
            raise RuntimeError("D455 returned no color frame")
        number = old.get_frame_number()
        captured = time.monotonic()
        source_frames=self.pipeline.wait_for_frames(1000)
        frames=self.align.process(source_frames)
        frame = frames.get_color_frame()
        if not frame or frame.get_frame_number() <= max(number, self.last_number):
            raise RuntimeError("Repeated or missing D455 color frame")
        self.last_number = frame.get_frame_number()
        depth=frames.get_depth_frame()
        self.last_depth=None
        if depth:
            self.last_depth=np.asanyarray(depth.get_data()).copy()*self.depth_scale
            self.last_color=np.asanyarray(frame.get_data()).copy()
            self.intrinsics=frame.profile.as_video_stream_profile().get_intrinsics()
            raw_depth=source_frames.get_depth_frame()
            self.last_pair={
                'depth_frame_number':raw_depth.get_frame_number(),
                'color_timestamp_ms':frame.get_timestamp(),
                'depth_timestamp_ms':raw_depth.get_timestamp(),
                'color_timestamp_domain':str(frame.get_frame_timestamp_domain()),
                'depth_timestamp_domain':str(raw_depth.get_frame_timestamp_domain()),
            }
        return frame,captured

    def capture_ring_depth(self):
        """Keep the latest original JPEG and add repeated aligned depths for thin tabletop parts."""
        depths=[];numbers=[];depth_numbers=[];began=time.monotonic()
        for i in range(5):
            if i==4:image,captured,number=self.capture()
            else:
                self._aligned_pair()
                number=self.last_number
            if self.last_depth is None:
                raise RuntimeError('Missing depth in stationary ring observation')
            depths.append(self.last_depth)
            numbers.append(number);depth_numbers.append(self.last_pair['depth_frame_number'])
        if len(set(depth_numbers))!=5 or len(set(numbers))!=5:
            raise RuntimeError('Repeated depth or color pair in stationary ring observation')
        from .ring_support_reference import window_depth
        self.ring_depth_window={'depth_m':window_depth(depths),'latest_pair_m':depths[-2:],
            'frame_numbers':numbers,'depth_frame_numbers':depth_numbers,'duration_s':time.monotonic()-began}
        return image,captured,number

    def save_depth(self, path):
        import numpy as np
        if self.last_depth is None:
            raise RuntimeError('No aligned depth for current camera frame')
        i=self.intrinsics
        window=getattr(self,'ring_depth_window',None)
        extras=({'ring_depth_median_m':window['depth_m'],'ring_latest_pair_m':window['latest_pair_m'],
                 'ring_color_frame_numbers':window['frame_numbers'],'ring_depth_frame_numbers':window['depth_frame_numbers']}
                if window else {})
        np.savez_compressed(path,depth_m=self.last_depth,bgr=self.last_color,
                            intrinsics=np.array([i.fx,i.fy,i.ppx,i.ppy]),frame_number=self.last_number,
                            K=np.array([[i.fx,0.,i.ppx],[0.,i.fy,i.ppy],[0.,0.,1.]]),
                            D=np.asarray(i.coeffs),distortion_model=str(i.model),
                            camera_serial=self.serial,depth_scale_m=self.depth_scale,
                            resolution=np.array([i.width,i.height]),**self.last_pair,**extras)

    def close(self):
        self.pipeline.stop()
