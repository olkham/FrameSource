from typing import Optional, Tuple, Any
import numpy as np
import cv2
import logging
try:
    from .video_capture_base import VideoCaptureBase
except ImportError:
    from video_capture_base import VideoCaptureBase
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoFileCapture(VideoCaptureBase):
    def start_async(self):
        """
        Start background thread to continuously capture frames from video file.
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
            Tuple[bool, Optional[np.ndarray]]: (success, frame) where success indicates if frame is available
        """
        frame = getattr(self, '_latest_frame', None)
        return (frame is not None), frame

    def _read_direct(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Directly read a frame from the video file (bypassing background thread logic).
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self.is_connected or self.cap is None:
            return False, None
        # Add delay for real-time playback simulation
        if self.real_time:
            video_fps = self.cap.get(cv2.CAP_PROP_FPS)
            if video_fps > 0:
                frame_duration = 1.0 / video_fps
                current_time = time.time()
                elapsed = current_time - self.time_of_last_frame
                if elapsed < frame_duration:
                    time.sleep(frame_duration - elapsed)
                self.time_of_last_frame = time.time()
        ret, frame = self.cap.read()
        # If we've reached the end of the video and looping is enabled
        if not ret and self.loop:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if self.real_time:
                self.time_of_last_frame = time.time()
        return ret, frame if ret else None

    """Video file capture using OpenCV."""
    
    def __init__(self, source: str, **kwargs):
        super().__init__(source, **kwargs)
        self.cap = None
        self.loop = kwargs.get('loop', False)
        self.real_time = kwargs.get('real_time', True)
        self.time_of_last_frame = 0.0
        
    def connect(self) -> bool:
        """Connect to video file."""
        try:
            self.cap = cv2.VideoCapture(self.source)
            if not self.cap.isOpened():
                logger.error(f"Failed to open video file {self.source}")
                return False
            
            # Set additional parameters if provided
            if 'width' in self.config:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['width'])
            if 'height' in self.config:
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['height'])
            if 'fps' in self.config:
                self.cap.set(cv2.CAP_PROP_FPS, self.config['fps'])
                
            self.is_connected = True
            logger.info(f"Connected to video file {self.source}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to video file: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from video file."""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.is_connected = False
            logger.info("Disconnected from video file")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from video file: {e}")
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
        """Set exposure (not applicable for video files)."""
        logger.warning("Exposure control not applicable for video files")
        return False
    
    def get_exposure(self) -> Optional[float]:
        """Get exposure (not applicable for video files)."""
        return None
    
    def set_gain(self, value: float) -> bool:
        """Set gain (not applicable for video files)."""
        logger.warning("Gain control not applicable for video files")
        return False
    
    def get_gain(self) -> Optional[float]:
        """Get gain (not applicable for video files)."""
        return None

    def enable_auto_exposure(self, enable: bool = True) -> bool:
        """
        Enable or disable auto exposure (not applicable for video files).
        
        Args:
            enable: True to enable, False to disable
        
        Returns:
            bool: Always False for video files
        """
        logger.warning("Auto exposure control not applicable for video files")
        return False
    
    def set_frame_size(self, width: int, height: int) -> bool:
        """Set frame size (not applicable for video files)."""
        logger.warning("Setting resolution is not applicable for video files")
        return False

    @classmethod
    def discover(cls) -> list:
        """
        Discover method for video file capture.
        
        Returns:
            list: Empty list, as discovery is not applicable for file-based sources.
                Use this class directly with file paths as the source parameter.
        """
        # Video file capture doesn't discover devices - it works with file paths
        logger.info("VideoFileCapture uses file paths as sources, not discoverable devices.")
        return []


if __name__ == "__main__":
    # Example usage
    video_file = "path/to/your/video.mp4"  # Replace with your video file path
    camera = VideoFileCapture(source=video_file, loop=True, real_time=True)
    
    if camera.connect():
        print("Webcam connected successfully.")
        print(f"Exposure: {camera.get_exposure()}")
        print(f"Gain: {camera.get_gain()}")
        print(f"Frame size: {camera.get_frame_size()}")
        
        # Read a few frames
        while camera.is_connected:
            ret, frame = camera.read()
            if ret and frame is not None:
                cv2.imshow("Webcam", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        camera.disconnect()
    else:
        print("Failed to connect to webcam.")