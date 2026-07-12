from typing import Optional, Tuple, Any, Dict
import numpy as np
import cv2
import logging
import warnings
from .video_capture_base import VideoCaptureBase
import time

logger = logging.getLogger(__name__)


class VideoFileCapture(VideoCaptureBase):
    has_discovery = False  # Uses file paths, not discoverable devices
    supports_exposure = False  # Video files have no controllable exposure
    supports_gain = False      # Video files have no controllable gain

    """Video file capture using OpenCV."""

    def __init__(self, source: str, *, loop: bool = False, real_time: bool = True,
                 width: Optional[int] = None, height: Optional[int] = None,
                 fps: Optional[float] = None, **kwargs):
        """Initialize the video file capture.

        Args:
            source: Path to the video file.
            loop: Restart playback from the beginning when the end is reached.
            real_time: Play at the file's native frame rate (disable for
                fastest processing).
            width: Optional target frame width (applied on :meth:`connect`).
            height: Optional target frame height (applied on :meth:`connect`).
            fps: Optional target frame rate (applied on :meth:`connect`).
            **kwargs: Additional passthrough options stored on ``self.config``.
        """
        # Forward the promoted options back into ``self.config`` so behaviour
        # that reads them (``connect()`` for width/height/fps) is preserved.
        for _key, _val in (('loop', loop), ('real_time', real_time),
                           ('width', width), ('height', height), ('fps', fps)):
            if _val is not None:
                kwargs[_key] = _val
        super().__init__(source, **kwargs)
        self.cap = None
        self.loop = loop
        self.real_time = real_time
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
        Read a single frame from the video file.
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

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Get configuration schema for video file capture"""
        warnings.warn(
            "get_config_schema() is deprecated and will be removed in a future release; "
            "UI form schemas belong in the consuming application.",
            DeprecationWarning,
            stacklevel=2,
        )
        return {
            'title': 'Video File Configuration',
            'description': 'Configure video file playback settings',
            'fields': [
                {
                    'name': 'source',
                    'label': 'Video File Path',
                    'type': 'text',
                    'placeholder': 'C:/path/to/video.mp4',
                    'description': 'Full path to video file (supports .mp4, .avi, .mov, .mkv, .wmv, .flv, .webm)',
                    'required': True
                },
                {
                    'name': 'loop',
                    'label': 'Loop Playback',
                    'type': 'checkbox',
                    'description': 'Restart video from beginning when it ends',
                    'required': False,
                    'default': False
                },
                {
                    'name': 'real_time',
                    'label': 'Real-time Playback',
                    'type': 'checkbox',
                    'description': 'Play video at original frame rate (disable for fastest processing)',
                    'required': False,
                    'default': True
                },
                {
                    'name': 'width',
                    'label': 'Width',
                    'type': 'number',
                    'min': 160,
                    'max': 4096,
                    'placeholder': '1920',
                    'description': 'Resize frame width (optional)',
                    'required': False
                },
                {
                    'name': 'height',
                    'label': 'Height',
                    'type': 'number',
                    'min': 120,
                    'max': 2160,
                    'placeholder': '1080',
                    'description': 'Resize frame height (optional)',
                    'required': False
                }
            ]
        }


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