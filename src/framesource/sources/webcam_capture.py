import logging
import platform
import warnings
from typing import Any, Optional

import cv2
import numpy as np

from ..discovery import DeviceInfo
from .video_capture_base import VideoCaptureBase

logger = logging.getLogger(__name__)


class WebcamCapture(VideoCaptureBase):
    has_discovery = True
    supports_exposure = True
    supports_gain = True

    """Webcam capture using OpenCV."""

    def __init__(
        self,
        source: int = 0,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        **kwargs,
    ):
        """Initialize the webcam capture.

        Args:
            source: Camera device index, or an ``api_pref:index`` /
                ``api_pref:path`` string (as produced in the ``id`` field of
                :meth:`discover`).
            width: Desired frame width in pixels. Applied on :meth:`connect`
                (together with ``height``) when provided.
            height: Desired frame height in pixels. Applied on
                :meth:`connect` (together with ``width``) when provided.
            fps: Desired frames per second. Applied on :meth:`connect` when
                provided.
            **kwargs: Additional passthrough options stored on ``self.config``.
        """
        # Preserve the historical ``self.config`` contents: these options were
        # previously mined from ``**kwargs`` and read back via ``self.config``
        # in ``connect()``. Only forward explicitly-provided values so the
        # ``'width' in self.config`` presence checks keep their old meaning.
        for _key, _val in (("width", width), ("height", height), ("fps", fps)):
            if _val is not None:
                kwargs[_key] = _val
        super().__init__(source, **kwargs)
        self.cap = None
        # Set API preference based on OS
        if platform.system() == "Windows":
            self.api_preference = cv2.CAP_DSHOW  # DirectShow for Windows
        elif platform.system() == "Darwin":
            self.api_preference = cv2.CAP_AVFOUNDATION  # AVFoundation for macOS
        else:
            self.api_preference = cv2.CAP_V4L2  # Video4Linux for Linux

        self.source = source

        if "is_mono" in kwargs:
            logger.warning(
                "'is_mono' argument is only used for certain industrial cameras "
                "and has no effect for webcams."
            )

    def connect(self) -> bool:
        """Connect to webcam."""
        try:
            src = self.source
            api_pref = self.api_preference
            # Support `api_pref/index` format or `api_pref/path` format used in
            # list_devices `id` field
            if isinstance(src, str) and ":" in src:
                parts = src.split(":")
                api_pref, src = parts[0], parts[1]

                if api_pref.isdigit():
                    api_pref = int(api_pref)

                if src.isdigit():
                    src = int(src)

            self.cap = cv2.VideoCapture(src, api_pref)
            if not self.cap.isOpened():
                logger.error(f"Failed to open webcam {src}")
                return False

            # Set additional parameters if provided
            if "width" in self.config and "height" in self.config:
                self.set_frame_size(self.config["width"], self.config["height"])
            if "fps" in self.config:
                self.cap.set(cv2.CAP_PROP_FPS, self.config["fps"])

            self.is_connected = True
            logger.info(f"Connected to webcam {src}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to webcam: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from webcam."""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.is_connected = False
            logger.info("Disconnected from webcam")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from webcam: {e}")
            return False

    def _read_implementation(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the webcam.
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self.is_connected or self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        return ret, frame if ret else None

    def set_exposure(self, value: float) -> bool:
        """Set exposure (-13 to -1 for most webcams)."""
        if not self.is_connected or self.cap is None:
            return False

        try:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, value)
            self._exposure = value
            return True
        except Exception as e:
            logger.error(f"Error setting exposure: {e}")
            return False

    def get_exposure(self) -> Optional[float]:
        """Get current exposure."""
        if not self.is_connected or self.cap is None:
            return None

        try:
            return self.cap.get(cv2.CAP_PROP_EXPOSURE)
        except Exception:
            return self._exposure

    def set_gain(self, value: float) -> bool:
        """Set gain (0-255 for most webcams)."""
        if not self.is_connected or self.cap is None:
            return False

        try:
            self.cap.set(cv2.CAP_PROP_GAIN, value)
            self._gain = value
            return True
        except Exception as e:
            logger.error(f"Error setting gain: {e}")
            return False

    def get_gain(self) -> Optional[float]:
        """Get current gain."""
        if not self.is_connected or self.cap is None:
            return None

        try:
            return self.cap.get(cv2.CAP_PROP_GAIN)
        except Exception:
            return self._gain

    def get_frame_size(self) -> Optional[tuple[int, int]]:
        """Get frame size."""
        if not self.is_connected or self.cap is None:
            return None

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)

    def set_frame_size(self, width: int, height: int) -> bool:
        """Set frame size."""
        if self.cap is None:
            return False
        result1 = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        result2 = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        logger.info(f"Set webcam resolution to {width}x{height} (success: {result1 and result2})")
        return result1 and result2

    def get_fps(self) -> Optional[float]:
        """Get FPS."""
        if not self.is_connected or self.cap is None:
            return None
        return self.cap.get(cv2.CAP_PROP_FPS)

    def set_fps(self, fps: float) -> bool:
        """Set FPS."""
        if not self.is_connected or self.cap is None:
            return False
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        return True

    def enable_auto_exposure(self, enable: bool = True) -> bool:
        """
        Enable or disable auto exposure for webcam.
        """
        if not self.is_connected or self.cap is None:
            return False
        try:
            # OpenCV expects 0.75 for auto, 0.25 for manual (on many webcams)
            value = 0.75 if enable else 0.25
            result = self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, value)
            logger.info(f"Set auto exposure to {enable} (cv2 value: {value})")
            return result
        except Exception as e:
            logger.error(f"Error setting auto exposure: {e}")
            return False

    @classmethod
    def discover(cls) -> list[DeviceInfo]:
        try:
            devices: list[DeviceInfo] = []
            from cv2.videoio_registry import getBackendName
            from cv2_enumerate_cameras import enumerate_cameras

            camera_list = []
            if platform.system() == "Windows":
                backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
            elif platform.system() == "Darwin":  # macOS
                backends = [cv2.CAP_AVFOUNDATION]
            else:  # Linux
                backends = [cv2.CAP_V4L2]

            for backend in backends:
                camera_list.extend(enumerate_cameras(backend))

            # Sort camera list by index
            camera_list.sort(key=lambda cam: cam.index)

            for camera_info in camera_list:
                # OS enumeration index -> not a stable identifier across replug.
                devices.append(
                    DeviceInfo(
                        device_id=f"{camera_info.backend}:{camera_info.index}:{camera_info.path}",
                        index=camera_info.index,
                        name=camera_info.name,
                        driver="opencv",
                        id_stable=False,
                        metadata={
                            "backend_index": camera_info.backend,
                            "backend_name": getBackendName(camera_info.backend),
                        },
                    )
                )
            logger.info(f"Found {cls.__name__} input device: {devices}")
            return devices
        except ImportError:
            logger.warning(
                "cv2-enumerate-cameras module not available. Install cv2-enumerate-cameras "
                "to list available (web)cameras."
            )
        return []

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Get configuration schema for webcam capture"""
        warnings.warn(
            "get_config_schema() is deprecated and will be removed in a future release; "
            "UI form schemas belong in the consuming application.",
            DeprecationWarning,
            stacklevel=2,
        )
        return {
            "title": "Webcam Configuration",
            "description": "Configure USB webcam or built-in camera settings",
            "fields": [
                {
                    "name": "source",
                    "label": "Camera Index",
                    "type": "number",
                    "min": 0,
                    "max": 10,
                    "placeholder": "0",
                    "description": "Camera device index (0 for default, 1 for second camera, etc.)",
                    "required": False,
                    "default": 0,
                },
                {
                    "name": "width",
                    "label": "Width",
                    "type": "number",
                    "min": 160,
                    "max": 4096,
                    "placeholder": "1920",
                    "description": "Frame width in pixels",
                    "required": False,
                },
                {
                    "name": "height",
                    "label": "Height",
                    "type": "number",
                    "min": 120,
                    "max": 2160,
                    "placeholder": "1080",
                    "description": "Frame height in pixels",
                    "required": False,
                },
                {
                    "name": "fps",
                    "label": "Frame Rate (FPS)",
                    "type": "number",
                    "min": 1,
                    "max": 120,
                    "placeholder": "30",
                    "description": "Frames per second",
                    "required": False,
                    "default": 30,
                },
                {
                    "name": "exposure",
                    "label": "Exposure",
                    "type": "number",
                    "min": -13,
                    "max": -1,
                    "step": 1,
                    "placeholder": "-6",
                    "description": "Manual exposure value (-13 to -1, lower = brighter)",
                    "required": False,
                },
                {
                    "name": "gain",
                    "label": "Gain",
                    "type": "number",
                    "min": 0,
                    "max": 255,
                    "placeholder": "0",
                    "description": "Camera gain (0-255)",
                    "required": False,
                },
            ],
        }


if __name__ == "__main__":
    # Example usage

    print("Discovering webcams...")
    webcams = WebcamCapture.discover()
    print(f"\nFound {len(webcams)} webcam(s):\n")
    for cam in webcams:
        print(f"  [{cam['index']}] {cam['name']}")

    if webcams:
        print("\nTesting first camera...")
        camera = WebcamCapture(source=0)
        if camera.connect():
            print("Webcam connected successfully.")
            print(f"Exposure: {camera.get_exposure()}")
            print(f"Gain: {camera.get_gain()}")
            print(f"Frame size: {camera.get_frame_size()}")

            # Read a few frames
            print("Press 'q' to quit...")
            while camera.is_connected:
                ret, frame = camera.read()
                if ret:
                    cv2.imshow("Webcam", frame)  # type: ignore
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    print("Failed to read frame from webcam.")
                    break

            camera.disconnect()
        else:
            print("Failed to connect to webcam.")
    else:
        print("\nNo webcams found.")
