from typing import Optional, Tuple, Any
import numpy as np
import cv2
import logging
from .video_capture_base import VideoCaptureBase
import platform
from frame_processors.realsense_depth_processor import RealsenseDepthProcessor, RealsenseProcessingOutput

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealsenseCapture(VideoCaptureBase):
    def start_async(self):
        """
        Start background thread to continuously capture frames from realsense camera.
        """
        import threading
        import time
        if hasattr(self, '_capture_thread') and self._capture_thread is not None and self._capture_thread.is_alive():
            return  # Already running
        self._stop_event = threading.Event()
        self._latest_frame = None
        self._capture_thread = threading.Thread(target=self._background_capture, daemon=True)
        self._capture_thread.start()

    def stop(self):
        """
        Stop background frame capture thread.
        """
        if hasattr(self, '_stop_event') and self._stop_event is not None:
            self._stop_event.set()
        if hasattr(self, '_capture_thread') and self._capture_thread is not None:
            self._capture_thread.join(timeout=2)
        self._capture_thread = None
        self._stop_event = None

    def _background_capture(self):
        import time
        while not self._stop_event.is_set():  # type: ignore
            success, frame = self._read_direct()
            if success:
                self._latest_frame = frame
            time.sleep(0.01)  # ~100 FPS max, adjust as needed

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the most recent frame captured by the background thread.
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        frame = getattr(self, '_latest_frame', None)
        self._latest_frame = None  # Clear after reading to avoid stale data
        return (frame is not None), frame

    def _read_direct(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Directly read a frame from the realsense camera (bypassing background thread logic).
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self.is_connected or self.pipeline is None:
            return False, None

        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()
        aligned = self._align.process(frames)

        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        aligned_depth_frame = aligned.get_depth_frame()
        aligned_color_frame = aligned.get_color_frame()

        if not depth_frame or not color_frame:
            return False, None

        # if depth_frame and color_frame:
        #     print("Depth Intrinsics:", depth_frame.profile.as_video_stream_profile().get_intrinsics())
        #     print("Color Intrinsics:", color_frame.profile.as_video_stream_profile().get_intrinsics())

        frame_dict = {
            'raw_color': color_frame,
            'raw_depth': depth_frame,
            'aligned_depth': aligned_depth_frame,
            'aligned_color': aligned_color_frame,
        }

        return True, frame_dict

    """Realsense camera capture using Realsense lib."""

    def __init__(self, source: int = 0, **kwargs):
        super().__init__(source, **kwargs)
        self.pipeline = None
        self.device = None
        self.profile = None
        self._align = None

        self.w = self.config.get("width", 0)
        self.h = self.config.get("height", 0)
        self.fps = self.config.get("fps", 0)

        self._max_width = 0
        self._max_height = 0
        self._max_fps = 0
        
        self._default_processor = self.config.get("processor", None)
        if self._default_processor is None:
            self._default_processor = RealsenseDepthProcessor(output_format=RealsenseProcessingOutput.RGB)
        self.attach_processor(self._default_processor)
        
        self.source = source if isinstance(source, int) else 0

        if 'is_mono' in kwargs:
            logger.warning("'is_mono' argument is only used for Ximea cameras and has no effect for realsense camera.")

    def connect(self) -> bool:
        """Connect to realsense camera."""
        try:
            import pyrealsense2 as rs

            # Configure depth and color streams
            self.pipeline = rs.pipeline()
            self._align = rs.align(rs.stream.color)
            config = rs.config()

            serial_number = self.source if isinstance(self.source, str) else None
            device_index = self.source if isinstance(self.source, int) else 0

            if serial_number:
                config.enable_device(serial_number)
            elif device_index is not None:
                ctx = rs.context()
                devices = ctx.query_devices()

                if device_index < 0 or device_index >= len(devices):
                    raise ValueError(f"Invalid device index: {device_index}")
                selected_serial = devices[device_index].get_info(rs.camera_info.serial_number)
                config.enable_device(selected_serial)

            # Get device product line for setting a supporting resolution
            pipeline_wrapper = rs.pipeline_wrapper(self.pipeline)
            pipeline_profile = config.resolve(pipeline_wrapper)
            self.device: rs.device = pipeline_profile.get_device()

            serial_number = self.device.get_info(rs.camera_info.serial_number)
            device_name = self.device.get_info(rs.camera_info.name)
            device_product_line = str(self.device.get_info(rs.camera_info.product_line))

            found_rgb = False
            max_res = (0, 0, 0)
            for s in self.device.query_sensors():
                if any(p.stream_type() == rs.stream.color for p in s.get_stream_profiles()):
                    found_rgb = True
                for profile in s.get_stream_profiles():
                    if profile.stream_type() == rs.stream.color:
                        vp = profile.as_video_stream_profile()
                        res = (vp.width(), vp.height(), vp.fps())
                        if res > max_res:
                            max_res = res
            if not found_rgb:
                print("The demo requires Depth camera with Color sensor")
                exit(0)

            self._max_width = max_res[0]
            self._max_height = max_res[1]
            self._max_fps = max_res[2]

            width = self.w if self.w > 0 else self._max_width
            height = self.h if self.h > 0 else self._max_height
            fps = self.fps if self.fps > 0 else self._max_fps

            config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

            # Start streaming
            self.profile = self.pipeline.start(config)

            self.is_connected = True
            logger.info(f"Connected to realsense camera {self.source}, {device_name} ({device_product_line} line) (Serial: {serial_number})")
            return True
        except Exception as e:
            logger.error(f"Error connecting to realsense camera: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from realsense camera."""
        try:
            if self.pipeline is not None:
                self.pipeline.stop()
            self.is_connected = False
            logger.info("Disconnected from realsense camera")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from realsense camera: {e}")
            return False

    def _read_implementation(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Return the latest frame captured by the background thread, or fall back to direct read if not running.
        """
        if hasattr(self, '_capture_thread') and self._capture_thread is not None and self._capture_thread.is_alive():
            return self.get_latest_frame()
        else:
            return self._read_direct()

    def set_exposure(self, value: float) -> bool:
        """Set exposure (1.0 to 165000 for most realsense cameras)."""
        if not self.is_connected or self.pipeline is None:
            return False

        try:
            self._exposure = value
            import pyrealsense2 as rs
            for s in self.device.query_sensors():
                if s.supports(rs.option.exposure):
                    s.set_option(rs.option.exposure, int(value))
                    logger.info(f"Set exposure to {int(value)}")
                else:
                    logger.info(f"Not setting exposure on sensor {s}, it is not supported.")

            return True
        except Exception as e:
            logger.error(f"Error setting exposure: {e}")
            return False

    def get_exposure(self) -> Optional[float]:
        """Get current exposure."""
        if not self.is_connected or self.pipeline is None:
            return None

        try:
            import pyrealsense2 as rs
            for s in self.device.query_sensors():
                if s.supports(rs.option.exposure):
                    exposure = s.get_option(rs.option.exposure)
                    return float(exposure)
        except Exception:
            return self._exposure

    def set_gain(self, value: float) -> bool:
        """Set gain (16-248 for most realsense cameras)."""
        if not self.is_connected or self.pipeline is None:
            return False

        try:
            self._gain = value
            import pyrealsense2 as rs
            for s in self.device.query_sensors():
                if s.supports(rs.option.gain):
                    s.set_option(rs.option.gain, int(value))
                    logger.info(f"Set gain to {int(value)}")
                else:
                    logger.info(f"Not setting gain on sensor {s}, it is not supported.")

            return True
        except Exception as e:
            logger.error(f"Error setting gain: {e}")
            return False

    def get_gain(self) -> Optional[float]:
        """Get current gain."""
        if not self.is_connected or self.pipeline is None:
            return None

        try:
            import pyrealsense2 as rs
            for s in self.device.query_sensors():
                if s.supports(rs.option.gain):
                    gain = s.get_option(rs.option.gain)
                    return float(gain)
        except Exception:
            return self._gain


    def get_exposure_range(self) -> Optional[Tuple[float, float]]:
        """Get current exposure."""
        if not self.is_connected or self.pipeline is None:
            return None

        try:
            import pyrealsense2 as rs
            for s in self.device.query_sensors():
                if s.supports(rs.option.exposure):
                    exposure = s.get_option_range(rs.option.exposure)
                    return float(exposure.min), float(exposure.max)
        except Exception:
            return 0.0, 0.0

    def get_gain_range(self) -> Optional[Tuple[float, float]]:
        """Get current gain."""
        if not self.is_connected or self.pipeline is None:
            return None

        try:
            import pyrealsense2 as rs
            for s in self.device.query_sensors():
                if s.supports(rs.option.gain):
                    gain = s.get_option_range(rs.option.gain)
                    return float(gain.min), float(gain.max)
        except Exception:
            return 0.0, 0.0

    def _get_active_profile(self):
        import pyrealsense2 as rs
        color_stream = self.profile.get_stream(rs.stream.color)
        video_profile = color_stream.as_video_stream_profile()
        return video_profile

    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        """Get frame size."""
        if not self.is_connected or self.profile is None:
            return None

        # Get current resolution
        video_profile = self._get_active_profile()
        width = video_profile.width()
        height = video_profile.height()

        return width, height

    def set_frame_size(self, width: int, height: int) -> bool:
        """Set frame size."""
        self.w = width
        self.h = height

        return True

    def get_fps(self) -> Optional[float]:
        """Get FPS."""
        if not self.is_connected or self.profile is None:
            return None

        video_profile = self._get_active_profile()
        fps = video_profile.fps()
        return fps

    def set_fps(self, fps: float) -> bool:
        self.fps = fps
        return True

    def enable_auto_exposure(self, enable: bool = True) -> bool:
        """
        Enable or disable auto exposure for realsense camera.
        """
        if not self.is_connected or self.pipeline is None:
            return False
        try:
            import pyrealsense2 as rs
            for s in self.device.query_sensors():
                # Check if the sensor supports auto-exposure
                if s.supports(rs.option.enable_auto_exposure):
                    s.set_option(rs.option.enable_auto_exposure, int(enable))
                    logger.info(f"Set auto exposure to {enable}")
                else:
                    logger.info(f"Not setting auto exposure on sensor {s}, it is not supported.")

            return True
        except Exception as e:
            logger.error(f"Error setting auto exposure: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    camera = RealsenseCapture(source=0)
    if camera.connect():
        camera.start_async()
        print("Realsense camera connected successfully.")
        print(f"Exposure: {camera.get_exposure()}")
        print(f"Gain: {camera.get_gain()}")
        print(f"Frame size: {camera.get_frame_size()}")

        # Read a few frames
        while camera.is_connected:
            ret, frame = camera.read()
            if ret:
                cv2.imshow("Realsense camera", frame)  # type: ignore
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        camera.stop()
        camera.disconnect()
    else:
        print("Failed to connect to realsense camera.")
