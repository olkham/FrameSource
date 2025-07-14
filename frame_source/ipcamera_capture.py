from typing import Optional, Tuple, Any
import numpy as np
import cv2
import logging
from .video_capture_base import VideoCaptureBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class IPCameraCapture(VideoCaptureBase):
    def start_async(self):
        """
        Start background thread to continuously capture frames from IP camera.
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
        while not self._stop_event.is_set(): # type: ignore
            success, frame = self._read_direct()
            if success:
                self._latest_frame = frame
            time.sleep(0.01)  # ~100 FPS max, adjust as needed

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the most recent frame captured by the background thread.
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame) or (False, None) if not available
        """
        frame = getattr(self, '_latest_frame', None)
        return (frame is not None), frame

    def _read_direct(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Directly read a frame from the IP camera (bypassing background thread logic).
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self.is_connected or self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        return ret, frame if ret else None

    """IP Camera capture using OpenCV with RTSP/HTTP streams."""
    
    def __init__(self, source: str, username: Optional[str] = None, password: Optional[str] = None, **kwargs):
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
        
    def _read_implementation(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Return the latest frame captured by the background thread, or fall back to direct read if not running.
        """
        if hasattr(self, '_capture_thread') and self._capture_thread is not None and self._capture_thread.is_alive():
            return self.get_latest_frame()
        else:
            return self._read_direct()
    
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
        logger.info(f"Set IP camera resolution to {width}x{height} (success: {result1 and result2})")
        return result1 and result2


if __name__ == "__main__":
    # Example usage
    camera = IPCameraCapture(source="rtsp://192.168.1.153:554/h264Preview_01_sub",
                                 username="admin", password="password")
    
    if camera.connect():
        camera.start_async()
        print("IP Camera connected successfully.")
        print(f"Exposure: {camera.get_exposure()}")
        print(f"Gain: {camera.get_gain()}")
        print(f"Frame size: {camera.get_frame_size()}")
        
        # Read a few frames
        while camera.is_connected:
            ret, frame = camera.read()
            if ret:
                cv2.imshow("IP Camera", frame) # type: ignore
                if cv2.waitKey(1000) & 0xFF == ord('q'):
                    break
        camera.stop()
        camera.disconnect()