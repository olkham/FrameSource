import numpy as np
import cv2
import time
from typing import Optional, Tuple, Any
import logging
import threading

try:
    from .video_capture_base import VideoCaptureBase
except ImportError:
    # If running as main script, try absolute import
    from video_capture_base import VideoCaptureBase

try:
    import mss
except ImportError:
    mss = None
    logging.warning("mss is not installed. Install it with 'pip install mss' to use ScreenCapture.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScreenCapture(VideoCaptureBase):
    """
    Capture class for grabbing frames from a region of the screen.
    Args:
        x (int): Top-left x coordinate
        y (int): Top-left y coordinate
        w (int): Width of region
        h (int): Height of region
        fps (float): Target FPS (default 30)
    """
    def __init__(self, x: int = 0, y: int = 0, w: int = 640, h: int = 480, fps: float = 30.0, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.fps = fps
        self.monitor = {"top": y, "left": x, "width": w, "height": h}
        self._thread_local = threading.local()
        self.is_connected = False
        self.time_of_last_frame = 0.0

    def start_async(self):
        """
        Start background frame capture in a separate thread.
        Continuously updates self._latest_frame and self._latest_success.
        """
        import threading
        import time
        if hasattr(self, '_capture_thread') and self._capture_thread and self._capture_thread.is_alive():
            return  # Already running
        self._stop_event = threading.Event()
        self._latest_frame = None
        self._latest_success = False
        def _capture_loop():
            while not self._stop_event.is_set(): # type: ignore
                success, frame = self._read_frame_for_thread()
                self._latest_success = success
                self._latest_frame = frame
                time.sleep(0.01)  # 10ms delay to avoid busy loop
        self._capture_thread = threading.Thread(target=_capture_loop, daemon=True)
        self._capture_thread.start()

    def stop(self):
        """
        Stop background frame capture thread.
        """
        if hasattr(self, '_stop_event') and self._stop_event:
            self._stop_event.set()
        if hasattr(self, '_capture_thread') and self._capture_thread:
            self._capture_thread.join(timeout=1)
        self._capture_thread = None
        self._stop_event = None

    def _read_frame_for_thread(self):
        """
        Internal: Calls the direct read method for background thread use (avoids recursion).
        """
        return self._read_direct()

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the most recent frame captured by the background thread.
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        return getattr(self, '_latest_success', False), getattr(self, '_latest_frame', None)

    def connect(self) -> bool:
        if mss is None:
            logger.error("mss is not installed. Cannot use ScreenCapture.")
            return False
        self.is_connected = True
        logger.info(f"ScreenCapture connected to region x={self.x}, y={self.y}, w={self.w}, h={self.h}")
        return True

    def disconnect(self) -> bool:
        self.is_connected = False
        logger.info("ScreenCapture disconnected.")
        return True

    def _read_implementation(self) -> Tuple[bool, Optional[np.ndarray]]:
        # If background thread is running, return latest frame
        if hasattr(self, '_capture_thread') and self._capture_thread is not None and self._capture_thread.is_alive():
            return self.get_latest_frame()
        else:
            return self._read_direct()

    def _get_sct(self):
        if not hasattr(self._thread_local, 'sct') or self._thread_local.sct is None:
            if mss is None:
                raise RuntimeError("mss is not installed. Cannot create screen capture.")
            self._thread_local.sct = mss.mss()
        return self._thread_local.sct

    def _read_direct(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_connected:
            return False, None
        sct = self._get_sct()
        # Real-time playback control
        if self.fps > 0:
            frame_duration = 1.0 / self.fps
            now = time.time()
            elapsed = now - self.time_of_last_frame
            if elapsed < frame_duration:
                time.sleep(frame_duration - elapsed)
            self.time_of_last_frame = time.time()
        img = np.array(sct.grab(self.monitor))
        # Convert BGRA to BGR
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return True, frame

    def set_exposure(self, value: float) -> bool:
        logger.warning("Exposure control not applicable for screen capture.")
        return False

    def get_exposure(self) -> Optional[float]:
        return None

    def set_gain(self, value: float) -> bool:
        logger.warning("Gain control not applicable for screen capture.")
        return False

    def get_gain(self) -> Optional[float]:
        return None

    def enable_auto_exposure(self, enable: bool = True) -> bool:
        logger.warning("Auto exposure control not applicable for screen capture.")
        return False

    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        return (self.w, self.h)

    def set_frame_size(self, width: int, height: int) -> bool:
        self.w = width
        self.h = height
        self.monitor["width"] = width
        self.monitor["height"] = height
        return True

    def get_fps(self) -> Optional[float]:
        return self.fps

    def set_fps(self, fps: float) -> bool:
        self.fps = fps
        return True

    @classmethod
    def discover(cls) -> list:
        """
        Discover available screen capture sources (monitors/displays).
        
        Returns:
            list: List of dictionaries containing screen information.
                Each dict contains: {'index': int, 'name': str, 'width': int, 'height': int, 'left': int, 'top': int}
        """
        devices = []
        
        if mss is None:
            logger.warning("mss module not available. Cannot discover screen sources.")
            return []
        
        try:
            with mss.mss() as sct:
                # Get all monitors (index 0 is typically all monitors combined)
                monitors = sct.monitors
                
                for i, monitor in enumerate(monitors):
                    device_data = {
                        'index': i,
                        'name': f"Monitor {i}" if i > 0 else "All Monitors",
                        'width': monitor['width'],
                        'height': monitor['height'],
                        'left': monitor['left'],
                        'top': monitor['top']
                    }
                    devices.append(device_data)
                    logger.info(f"Found screen source: {device_data}")
                    
        except Exception as e:
            logger.error(f"Error discovering screen sources: {e}")
        
        return devices


if __name__ == "__main__":
    # Example usage
    
    screens = ScreenCapture.discover()
    print("Discovered screen sources:")
    for screen in screens:
        print(f" - {screen['name']} (#{screen['index']}): {screen['width']}x{screen['height']}")

    camera = ScreenCapture(x=100, y=100, w=800, h=600, fps=30)
    if camera.connect():
        camera.start_async()
        print("Webcam connected successfully.")
        print(f"Exposure: {camera.get_exposure()}")
        print(f"Gain: {camera.get_gain()}")
        print(f"Frame size: {camera.get_frame_size()}")
        
        # Read a few frames
        while camera.is_connected:
            ret, frame = camera.read()
            if ret:
                cv2.imshow("Webcam", frame) # type: ignore
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        camera.stop()
        camera.disconnect()
    else:
        print("Failed to connect to webcam.")