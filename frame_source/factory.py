"""
Video Capture System with Factory Pattern

A comprehensive video capture system that supports multiple backends:
- Webcam (OpenCV)
- IP Camera (RTSP/HTTP)
- Ximea cameras
- Custom capture APIs

Usage:
    capture = FrameSourceFactory.create('webcam', source=0)
    capture.connect()
    capture.set_exposure(50)
    frame = capture.read()
"""

from typing import Any
import logging

from .genicam_capture import GenicamCapture
from .video_capture_base import VideoCaptureBase
from .basler_capture import BaslerCapture
from .ximea_capture import XimeaCapture
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

    _capture_types = {
        'folder': FolderCapture,
        'video_file': VideoFileCapture,
        'webcam': WebcamCapture,
        'ipcam': IPCameraCapture,
        'ximea': XimeaCapture,
        'basler': BaslerCapture,
        'screen': ScreenCapture,
        'genicam': GenicamCapture,
        'audio_spectrogram': AudioSpectrogramCapture
    }
    
    @classmethod
    def create(cls, capture_type: str, source: Any = None, **kwargs) -> VideoCaptureBase:
        """
        Create a video capture instance.
        
        Args:
            capture_type: Type of capture ('webcam', 'ipcam', 'ximea', 'custom')
            source: Source identifier
            **kwargs: Additional parameters for the specific capture type
            
        Returns:
            VideoCaptureBase: Configured capture instance
            
        Raises:
            ValueError: If capture_type is not supported
        """
        if capture_type not in cls._capture_types:
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
    def get_available_types(cls) -> list:
        """Get list of available capture types."""
        return list(cls._capture_types.keys())



