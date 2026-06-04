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
import importlib
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from .sources.video_capture_base import VideoCaptureBase

# Import capture classes with graceful handling for missing optional
# dependencies (e.g. vendor SDKs). Each entry maps a factory key to its module
# and class name; sources that fail to import are simply omitted.
_OPTIONAL_SOURCES = [
    ('webcam', 'webcam_capture', 'WebcamCapture'),
    ('ipcam', 'ipcamera_capture', 'IPCameraCapture'),
    ('basler', 'basler_capture', 'BaslerCapture'),
    ('genicam', 'genicam_capture', 'GenicamCapture'),
    ('realsense', 'realsense_capture', 'RealsenseCapture'),
    ('ximea', 'ximea_capture', 'XimeaCapture'),
    ('huateng', 'huateng_capture', 'HuatengCapture'),
    ('video_file', 'video_file_capture', 'VideoFileCapture'),
    ('folder', 'folder_capture', 'FolderCapture'),
    ('screen', 'screen_capture', 'ScreenCapture'),
    ('audio_spectrogram', 'audiospectrogram_capture', 'AudioSpectrogramCapture'),
]

_capture_imports: Dict[str, type] = {}

for _key, _module_name, _class_name in _OPTIONAL_SOURCES:
    try:
        _module = importlib.import_module(f'.sources.{_module_name}', __package__)
        _capture_imports[_key] = getattr(_module, _class_name)
    except Exception as e:  # noqa: BLE001 - vendor SDKs may raise non-ImportError
        logger.debug("%s unavailable: %s", _class_name, e)


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
        'ximea',
        'huateng',
        'audio_spectrogram'
    ]

    CameraSource = Literal[
        'webcam',
        'realsense',
        'genicam',
        'basler',
        'ximea',
        'huateng'
    ]

    _capture_types: Dict[str, type] = _capture_imports


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
            capture_type = kwargs.pop('capture_type', None)
        
        if not capture_type or capture_type not in cls._capture_types:
            available_types = ', '.join(cls._capture_types.keys())
            raise ValueError(f"Unsupported capture type: {capture_type}. Available types: {available_types}")
        
        if source is None:
            source = kwargs.pop('source', None)

        if source is None:
            Warning("Source not provided, defaulting to 0")

        capture_class = cls._capture_types[capture_type]
        cc = capture_class(source=source, **kwargs)

        connect = kwargs.pop('connect', True)

        if connect and cc is not None:
            cc.connect()

        return cc

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


if __name__ == "__main__":
    # Simple test to demonstrate functionality
    print("FrameSourceFactory Test")
    print("=" * 80)
    
    # Test 1: Import the package
    print("\n1️⃣ Testing package import...")
    try:
        import framesource
        print(f"   ✅ Package imported successfully")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        exit(1)
    
    # Test 2: Check FrameSourceFactory available types
    print("\n2⃣ Testing FrameSourceFactory available types...")
    try:
        from framesource import FrameSourceFactory
        available_types = FrameSourceFactory.get_available_types()
        print(f"   ✅ Available types: {available_types}")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        exit(1)
