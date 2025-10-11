from .factory import FrameSourceFactory
from .threading_utils import simple_frame_producer, FrameProducer, multiprocess_frame_producer, create_producer_consumer_pair

from .video_capture_base import VideoCaptureBase

# Import capture classes with error handling for missing dependencies
try:
    from .webcam_capture import WebcamCapture
except ImportError as e:
    WebcamCapture = None
    print(f"⚠️ WebcamCapture unavailable: {e}")

try:
    from .audiospectrogram_capture import AudioSpectrogramCapture
except ImportError as e:
    AudioSpectrogramCapture = None
    print(f"⚠️ AudioSpectrogramCapture unavailable: {e}")

try:
    from .basler_capture import BaslerCapture
except ImportError as e:
    BaslerCapture = None
    print(f"⚠️ BaslerCapture unavailable: {e}")

try:
    from .genicam_capture import GenicamCapture
except ImportError as e:
    GenicamCapture = None
    print(f"⚠️ GenicamCapture unavailable: {e}")

try:
    from .ipcamera_capture import IPCameraCapture
except ImportError as e:
    IPCameraCapture = None
    print(f"⚠️ IPCameraCapture unavailable: {e}")

try:
    from .video_file_capture import VideoFileCapture
except ImportError as e:
    VideoFileCapture = None
    print(f"⚠️ VideoFileCapture unavailable: {e}")

try:
    from .folder_capture import FolderCapture
except ImportError as e:
    FolderCapture = None
    print(f"⚠️ FolderCapture unavailable: {e}")

try:
    from .realsense_capture import RealsenseCapture
except ImportError as e:
    RealsenseCapture = None
    print(f"⚠️ RealsenseCapture unavailable: {e}")

try:
    from .screen_capture import ScreenCapture
except ImportError as e:
    ScreenCapture = None
    print(f"⚠️ ScreenCapture unavailable: {e}")


def get_available_sources():
    """Get list of available frame source types with metadata.
    
    Dynamically builds the list based on successfully imported capture classes.
    """
    # Define all possible sources with their metadata
    # Only sources with successfully imported classes will be included
    all_possible_sources = [
        {
            'type': 'webcam',
            'name': 'Webcam',
            'description': 'Built-in or USB camera',
            'icon': 'fas fa-video',
            'class': WebcamCapture,
            'primary': True
        },
        {
            'type': 'ip_camera',
            'name': 'IP Camera',
            'description': 'Network camera stream',
            'icon': 'fas fa-wifi',
            'class': IPCameraCapture,
            'primary': True
        },
        {
            'type': 'basler',
            'name': 'Basler',
            'description': 'Industrial camera',
            'icon': 'fas fa-camera',
            'class': BaslerCapture,
            'primary': True
        },
        {
            'type': 'realsense',
            'name': 'RealSense',
            'description': '3D depth camera',
            'icon': 'fas fa-cube',
            'class': RealsenseCapture,
            'primary': True
        },
        {
            'type': 'video_file',
            'name': 'Video File',
            'description': 'Pre-recorded video',
            'icon': 'fas fa-file-video',
            'class': VideoFileCapture,
            'primary': True
        },
        {
            'type': 'image_folder',
            'name': 'Image Folder',
            'description': 'Batch process images',
            'icon': 'fas fa-folder-open',
            'class': FolderCapture,
            'primary': True
        },
        {
            'type': 'screen',
            'name': 'Screen Capture',
            'description': 'Live desktop region',
            'icon': 'fas fa-desktop',
            'class': ScreenCapture,
            'primary': False
        },
        {
            'type': 'audio_spectrogram',
            'name': 'Audio Spectrogram',
            'description': 'Real-time audio visualization',
            'icon': 'fas fa-music',
            'class': AudioSpectrogramCapture,
            'primary': False
        },
        {
            'type': 'genicam',
            'name': 'GenICam',
            'description': 'Generic camera interface',
            'icon': 'fas fa-camera',
            'class': GenicamCapture,
            'primary': False
        }
    ]
    
    # Filter to only include sources where the class was successfully imported
    source_metadata = [
        source for source in all_possible_sources 
        if source['class'] is not None
    ]
    
    # Filter to only include sources where the class was successfully imported
    source_metadata = [
        source for source in all_possible_sources 
        if source['class'] is not None
    ]
    
    # Get available types from FrameSourceFactory
    available_factory_types = FrameSourceFactory.get_available_types()
    
    # Map UI types to factory types
    ui_to_factory_mapping = {
        'ip_camera': 'ipcam',
        'image_folder': 'folder',
        'video_file': 'video_file',
        'audio_spectrogram': 'audio_spectrogram'
    }
    
    # Build the list of available sources with full metadata
    available_sources = []
    for source in source_metadata:
        try:
            # Map UI type to factory type
            factory_type = ui_to_factory_mapping.get(source['type'], source['type'])
            
            # Check if the factory type is available
            if factory_type in available_factory_types:
                # Try to create an instance to verify dependencies
                test_instance = FrameSourceFactory.create(factory_type, source='test')
                
                # Get configuration schema from the class
                try:
                    config_schema = source['class'].get_config_schema()
                except Exception as e:
                    config_schema = {
                        'fields': [],
                        'error': f"Schema error: {str(e)}"
                    }
                
                available_sources.append({
                    'type': source['type'],
                    'name': source['name'],
                    'description': source['description'],
                    'icon': source['icon'],
                    'primary': source['primary'],
                    'available': True,
                    'factory_type': factory_type,
                    'config_schema': config_schema
                })
            else:
                # Source class imported but not in factory (shouldn't happen normally)
                available_sources.append({
                    'type': source['type'],
                    'name': source['name'],
                    'description': source['description'],
                    'icon': source['icon'],
                    'primary': source['primary'],
                    'available': False,
                    'error': f'Source type "{factory_type}" not available in FrameSourceFactory',
                    'config_schema': {
                        'fields': [],
                        'error': f'Source unavailable: {factory_type} not in factory'
                    }
                })
                
        except Exception as e:
            # Mark as unavailable if there are runtime issues
            available_sources.append({
                'type': source['type'],
                'name': source['name'],
                'description': source['description'],
                'icon': source['icon'],
                'primary': source['primary'],
                'available': False,
                'error': str(e),
                'config_schema': {
                    'fields': [],
                    'error': f'Source unavailable: {str(e)}'
                }
            })

    return available_sources


__all__ = [
    'FrameSourceFactory',
    'simple_frame_producer',
    'FrameProducer', 
    'multiprocess_frame_producer',
    'create_producer_consumer_pair',
    'get_available_sources'
]