import logging
import warnings
from typing import Any, Optional

import cv2
import numpy as np

from .video_capture_base import VideoCaptureBase

logger = logging.getLogger(__name__)


class IPCameraCapture(VideoCaptureBase):
    has_discovery = False  # Requires manual URL/credential configuration
    supports_exposure = False  # set_exposure() is a no-op stub (returns False)
    supports_gain = False  # set_gain() is a no-op stub (returns False)

    """IP Camera capture using OpenCV with RTSP/HTTP streams."""

    def __init__(
        self,
        source: str,
        username: Optional[str] = None,
        password: Optional[str] = None,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        **kwargs,
    ):
        """Initialize the IP camera capture.

        Args:
            source: RTSP/HTTP stream URL.
            username: Optional login username; when given together with
                ``password`` it is injected into the stream URL.
            password: Optional login password (see ``username``).
            width: Optional frame width hint (if supported by the camera),
                stored on ``self.config``.
            height: Optional frame height hint (see ``width``).
            fps: Optional expected frame rate (informational), stored on
                ``self.config``.
            **kwargs: Additional passthrough options stored on ``self.config``.
        """
        # Forward the promoted options into ``self.config`` exactly as before
        # (they were previously accepted via ``**kwargs``); only include
        # explicitly-provided values so config keys are unchanged.
        for _key, _val in (("width", width), ("height", height), ("fps", fps)):
            if _val is not None:
                kwargs[_key] = _val
        super().__init__(source, **kwargs)
        self.username = username
        self.password = password
        self.cap = None
        if username is not None and password is not None:
            self.stream_url = self._build_stream_url()
        else:
            self.stream_url = source

    def _build_stream_url(self) -> str:
        """Build stream URL with authentication if provided."""
        if self.username and self.password:
            # Insert credentials into URL
            if "://" in self.source:
                protocol, rest = self.source.split("://", 1)
                return f"{protocol}://{self.username}:{self.password}@{rest}"
        return self.source

    def connect(self) -> bool:
        """Connect to IP camera."""
        try:
            self.cap = cv2.VideoCapture(self.stream_url)
            if not self.cap.isOpened():
                logger.error(f"Failed to open IP camera stream: {self.stream_url}")
                return False

            # Set buffer size to reduce latency
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            self.is_connected = True
            logger.info(f"Connected to IP camera: {self.source}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to IP camera: {e}")
            return False

    def disconnect(self) -> bool:
        """Disconnect from IP camera."""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.is_connected = False
            logger.info("Disconnected from IP camera")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from IP camera: {e}")
            return False

    def _read_implementation(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the IP camera.
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self.is_connected or self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        return ret, frame if ret else None

    def set_exposure(self, value: float) -> bool:
        """Set exposure (may not be supported by all IP cameras)."""
        logger.warning("Exposure control may not be supported by IP cameras")
        self._exposure = value
        return False

    def get_exposure(self) -> Optional[float]:
        """Get exposure."""
        return self._exposure

    def set_gain(self, value: float) -> bool:
        """Set gain (may not be supported by all IP cameras)."""
        logger.warning("Gain control may not be supported by IP cameras")
        self._gain = value
        return False

    def get_gain(self) -> Optional[float]:
        """Get gain."""
        return self._gain

    def enable_auto_exposure(self, enable: bool = True) -> bool:
        """
        Enable or disable auto exposure (not generally supported for IP cameras).
        """
        logger.warning("Auto exposure control may not be supported by IP cameras")
        return False

    def set_frame_size(self, width: int, height: int) -> bool:
        """Set frame size (may not be supported by all IP cameras)."""
        if not self.is_connected or self.cap is None:
            return False
        result1 = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        result2 = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        logger.info(
            f"Set IP camera resolution to {width}x{height} (success: {result1 and result2})"
        )
        return result1 and result2

    @classmethod
    def discover(cls) -> list:
        """
        Discover method for IP camera capture.

        Returns:
            list: Empty list, as IP cameras require manual configuration with URLs/credentials.
                Use this class directly with RTSP/HTTP URLs as the source parameter.
        """
        # IP cameras require manual configuration - cannot be auto-discovered easily
        logger.info("IPCameraCapture requires manual configuration with URLs and credentials.")
        return []

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Get configuration schema for IP camera capture"""
        warnings.warn(
            "get_config_schema() is deprecated and will be removed in a future release; "
            "UI form schemas belong in the consuming application.",
            DeprecationWarning,
            stacklevel=2,
        )
        return {
            "title": "IP Camera Configuration",
            "description": "Configure IP camera with RTSP/HTTP stream settings",
            "fields": [
                {
                    "name": "source",
                    "label": "Stream URL",
                    "type": "text",
                    "placeholder": "rtsp://192.168.1.100:554/stream1",
                    "description": "RTSP or HTTP stream URL",
                    "required": True,
                },
                {
                    "name": "username",
                    "label": "Username",
                    "type": "text",
                    "placeholder": "admin",
                    "description": "Camera login username (optional)",
                    "required": False,
                },
                {
                    "name": "password",
                    "label": "Password",
                    "type": "password",
                    "placeholder": "password",
                    "description": "Camera login password (optional)",
                    "required": False,
                },
                {
                    "name": "width",
                    "label": "Width",
                    "type": "number",
                    "min": 160,
                    "max": 4096,
                    "placeholder": "1920",
                    "description": "Frame width in pixels (if supported)",
                    "required": False,
                },
                {
                    "name": "height",
                    "label": "Height",
                    "type": "number",
                    "min": 120,
                    "max": 2160,
                    "placeholder": "1080",
                    "description": "Frame height in pixels (if supported)",
                    "required": False,
                },
                {
                    "name": "fps",
                    "label": "Frame Rate (FPS)",
                    "type": "number",
                    "min": 1,
                    "max": 60,
                    "placeholder": "25",
                    "description": "Expected frame rate (informational)",
                    "required": False,
                    "default": 25,
                },
            ],
        }


if __name__ == "__main__":
    # Example usage
    camera = IPCameraCapture(
        source="rtsp://192.168.1.153:554/h264Preview_01_sub", username="admin", password="password"
    )

    if camera.connect():
        print("IP Camera connected successfully.")
        print(f"Exposure: {camera.get_exposure()}")
        print(f"Gain: {camera.get_gain()}")
        print(f"Frame size: {camera.get_frame_size()}")

        # Read a few frames
        while camera.is_connected:
            ret, frame = camera.read()
            if ret:
                cv2.imshow("IP Camera", frame)  # type: ignore
                if cv2.waitKey(1000) & 0xFF == ord("q"):
                    break
        camera.stop()
        camera.disconnect()
