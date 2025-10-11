"""
Video Capture System with Factory Pattern

A comprehensive video capture system that supports multiple backends:
- Webcam (OpenCV)
- IP Camera (RTSP/HTTP)
- Industrial cameras (Basler, GenICam)
- Custom capture APIs

Usage:
    capture = FrameSourceFactory.create('webcam', source=0)
    capture.connect()
    capture.set_exposure(50)
    frame = capture.read()
"""

from typing import Any, Dict, List, Optional, Literal
import logging

from .video_capture_base import VideoCaptureBase

# Import capture classes with error handling for missing dependencies
_capture_imports = {}

try:
    from .webcam_capture import WebcamCapture
    _capture_imports['webcam'] = WebcamCapture
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"WebcamCapture unavailable: {e}")

try:
    from .ipcamera_capture import IPCameraCapture
    _capture_imports['ipcam'] = IPCameraCapture
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"IPCameraCapture unavailable: {e}")

try:
    from .basler_capture import BaslerCapture
    _capture_imports['basler'] = BaslerCapture
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"BaslerCapture unavailable: {e}")

try:
    from .genicam_capture import GenicamCapture
    _capture_imports['genicam'] = GenicamCapture
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"GenicamCapture unavailable: {e}")

try:
    from .realsense_capture import RealsenseCapture
    _capture_imports['realsense'] = RealsenseCapture
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"RealsenseCapture unavailable: {e}")

try:
    from .video_file_capture import VideoFileCapture
    _capture_imports['video_file'] = VideoFileCapture
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"VideoFileCapture unavailable: {e}")

try:
    from .folder_capture import FolderCapture
    _capture_imports['folder'] = FolderCapture
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"FolderCapture unavailable: {e}")

try:
    from .screen_capture import ScreenCapture
    _capture_imports['screen'] = ScreenCapture
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"ScreenCapture unavailable: {e}")

try:
    from .audiospectrogram_capture import AudioSpectrogramCapture
    _capture_imports['audio_spectrogram'] = AudioSpectrogramCapture
except ImportError as e:
    logger = logging.getLogger(__name__)
    logger.warning(f"AudioSpectrogramCapture unavailable: {e}")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FrameSourceFactory:
    """Factory class for creating video capture instances."""

    MediaSource = Literal[
        'folder',
        'video_file',
        'webcam',
        'ipcam',
        'basler',
        'realsense',
        'screen',
        'genicam',
        'audio_spectrogram'
    ]

    CameraSource = Literal[
        'webcam',
        'realsense',
        'genicam',
        'basler'
    ]

    _capture_types = {
        'folder': FolderCapture,
        'video_file': VideoFileCapture,
        'webcam': WebcamCapture,
        'ipcam': IPCameraCapture,
        'basler': BaslerCapture,
        'realsense': RealsenseCapture,
        'screen': ScreenCapture,
        'genicam': GenicamCapture,
        'audio_spectrogram': AudioSpectrogramCapture
    }

    @classmethod
    def create(cls, capture_type: Any = None, source: Any = None, **kwargs) -> VideoCaptureBase:
        """
        Create a video capture instance.
        
        Args:
            capture_type: Type of capture ('webcam', 'ipcam', 'basler', 'genicam', 'custom')
            source: Source identifier
            **kwargs: Additional parameters for the specific capture type
            
        Returns:
            VideoCaptureBase: Configured capture instance
            
        Raises:
            ValueError: If capture_type is not supported
        """
        # If capture_type is not provided, try to get it from kwargs
        if not capture_type:
            capture_type = kwargs.pop('capture_type', '')
        
        if not capture_type or capture_type not in cls._capture_types:
            available_types = ', '.join(cls._capture_types.keys())
            raise ValueError(f"Unsupported capture type: {capture_type}. Available types: {available_types}")
        
        capture_class = cls._capture_types[capture_type]
        return capture_class(source=source, **kwargs)

    
    @classmethod
    def register_capture_type(cls, name: str, capture_class: type):
        """
        Register a new capture type.
        
        Args:
            name: Name of the capture type
            capture_class: Class implementing VideoCaptureBase
        """
        if not issubclass(capture_class, VideoCaptureBase):
            raise ValueError("Capture class must inherit from VideoCaptureBase")
        
        if name in cls._capture_types:
            logger.warning(f"Capture type '{name}' already registered, replacing with new class.")  

        cls._capture_types[name] = capture_class
        logger.info(f"Registered new capture type: {name}")

    @classmethod
    def discover_devices(cls, sources: Optional[List[str]] = None) -> Dict:
        """
        Discover available capture devices from the registered capture types.

        This method queries each capture type for connected devices and
        returns a dictionary mapping source names to their discovery results.

        Args:
            sources (list[str], optional): Specific source keys to limit discovery to.
                If None, all registered capture types are queried.

        Returns:
            dict: A mapping of source keys to the discovered devices for each.
                  Sources that return no devices are excluded.
        """

        _sources = (
            cls._capture_types.items()
            if sources is None
            else ((k, v) for k, v in cls._capture_types.items() if k in sources)
        )

        return {
            k: ret
            for k, v in _sources
            if (ret := v.discover())
        }

    @classmethod
    def get_available_types(cls) -> list:
        """Get list of available capture types."""
        return list(cls._capture_types.keys())

    @classmethod
    def unregister_capture_type(cls, capture_type: str):
        """Unregister a capture type (convenience function)"""
        if capture_type not in cls._capture_types:
            raise ValueError(f"Capture type '{capture_type}' is not registered")
        del cls._capture_types[capture_type]
        logger.info(f"Unregistered capture type: {capture_type}")
