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

from .genicam_capture import GenicamCapture
from .realsense_capture import RealsenseCapture
from .video_capture_base import VideoCaptureBase
from .basler_capture import BaslerCapture
from .webcam_capture import WebcamCapture
from .ipcamera_capture import IPCameraCapture
from .video_file_capture import VideoFileCapture
from .folder_capture import FolderCapture
from .screen_capture import ScreenCapture
from .audiospectrogram_capture import AudioSpectrogramCapture

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
