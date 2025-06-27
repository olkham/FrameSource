"""
Video Capture System with Factory Pattern

A comprehensive video capture system that supports multiple backends:
- Webcam (OpenCV)
- IP Camera (RTSP/HTTP)
- Ximea cameras
- Custom capture APIs

Usage:
    capture = VideoCaptureFactory.create('webcam', source=0)
    capture.connect()
    capture.set_exposure(50)
    frame = capture.read()
"""

import cv2
from typing import Any, List
import logging
from video_capture_base import VideoCaptureBase
from basler_capture import BaslerCapture  # Import at the top level to avoid circular import inside class
from ximea_capture import XimeaCapture  # Import at the top level to avoid circular import inside class
from webcam_capture import WebcamCapture  # Import at the top level to avoid circular import inside class
from ipcamera_capture import IPCameraCapture  # Import at the top level to avoid circular import inside class
from video_file_capture import VideoFileCapture  # Import at the top level to avoid circular import inside class
from folder_capture import FolderCapture  # Import at the top level to avoid circular import inside class

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VideoCaptureFactory:
    """Factory class for creating video capture instances."""

    _capture_types = {
        'webcam': WebcamCapture,
        'ipcam': IPCameraCapture,
        'ximea': XimeaCapture,
        'video_file': VideoFileCapture,
        'basler': BaslerCapture,
        'folder': FolderCapture
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
        
        cls._capture_types[name] = capture_class
        logger.info(f"Registered new capture type: {name}")
    
    @classmethod
    def get_available_types(cls) -> list:
        """Get list of available capture types."""
        return list(cls._capture_types.keys())


def test_camera(name, **kwargs):
    # Example 1: Webcam capture
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    print("Testing Webcam Capture:")
    camera = VideoCaptureFactory.create(name, **kwargs)
    
    camera.connect()

    if camera.is_connected:

        exposure_range = camera.get_exposure_range()
        if exposure_range is not None:
            min_exp, max_exp = exposure_range
        else:
            min_exp, max_exp = None, None
            
        gain_range = camera.get_gain_range()
        if gain_range is not None:
            min_gain, max_gain = gain_range
        else:
            min_gain, max_gain = None, None

        camera.set_exposure(10000)
        camera.set_gain(0)

        # Lock exposure time but allow gain to vary for auto exposure
        try:
            # Disable full auto exposure/gain first
            # camera.enable_auto_exposure(True)
            
            # Set a fixed exposure time (10ms = 10000 microseconds)
            camera.set_exposure(10000)
            
            # Enable auto gain only while keeping exposure fixed
            camera.enable_auto_exposure(True)  # Enable auto exposure/gain
            
            print("Auto exposure/gain configured: exposure locked, gain variable")
        except Exception as e:
            print(f"Error configuring Ximea auto exposure/gain: {e}")

        # camera.enable_auto_exposure(True)
        print(f"Exposure: {camera.get_exposure()}")
        print(f"Gain: {camera.get_gain()}")
        print(f"Frame size: {camera.get_frame_size()}")
        
        # Read a few frames
        while camera.is_connected:
            ret, frame = camera.read()
            if ret:
                if frame is not None:
                    cv2.imshow("camera", frame)
                # Add key controls for exposure and gain adjustment
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('=') or key == ord('+'):  # Increase exposure
                    current_exposure = camera.get_exposure()
                    if current_exposure is not None and min_exp is not None and max_exp is not None:
                        new_exposure = min(current_exposure + 1000, max_exp)  # Increase by 1ms
                        camera.set_exposure(new_exposure)
                        print(f"Exposure increased to: {new_exposure} (range: {min_exp}-{max_exp})")
                elif key == ord('-'):  # Decrease exposure
                    current_exposure = camera.get_exposure()
                    if current_exposure is not None and min_exp is not None and max_exp is not None:
                        new_exposure = max(current_exposure - 1000, min_exp)  # Decrease by 1ms
                        camera.set_exposure(new_exposure)
                        print(f"Exposure decreased to: {new_exposure} (range: {min_exp}-{max_exp})")
                elif key == ord(']'):  # Increase gain
                    current_gain = camera.get_gain()
                    if current_gain is not None and min_gain is not None and max_gain is not None:
                        new_gain = min(current_gain + 1, max_gain)
                        camera.set_gain(new_gain)
                        print(f"Gain increased to: {new_gain} (range: {min_gain}-{max_gain})")
                elif key == ord('['):  # Decrease gain
                    current_gain = camera.get_gain()
                    if current_gain is not None and min_gain is not None and max_gain is not None:
                        new_gain = max(current_gain - 1, min_gain)
                        camera.set_gain(new_gain)
                        print(f"Gain decreased to: {new_gain} (range: {min_gain}-{max_gain})")
                elif key == ord('a'):  # Toggle auto exposure
                    print("Toggling auto exposure...")
                    camera.enable_auto_exposure(True)
                elif key == ord('m'):  # Manual exposure mode
                    print("Switching to manual exposure...")
                    camera.enable_auto_exposure(False)
                elif key == ord('h'):  # Show help
                    print("\nKey controls:")
                    print("  q - Quit")
                    print("  + or = - Increase exposure")
                    print("  - - Decrease exposure")
                    print("  ] - Increase gain")
                    print("  [ - Decrease gain")
                    print("  a - Enable auto exposure")
                    print("  m - Manual exposure mode")
                    print("  h - Show this help")
            else:
                print(f"Failed to read frame")

    camera.disconnect()


def test_multiple_cameras(cameras:List[Any], threaded:bool = True):
    """Test connecting to multiple different cameras types and viewing them live concurrently."""
    
    capture_instances = []
    for cam_cfg in cameras:
        name = cam_cfg.pop('capture_type', None)
        if not name:
            print(f"Camera config missing 'capture_type': {cam_cfg}")
            continue
        cv2.namedWindow(f"{name}", cv2.WINDOW_NORMAL)
        print(f"Testing {name} Capture:")
        camera = VideoCaptureFactory.create(name, **cam_cfg)
        if camera.connect():
            if threaded:
                camera.start()  # Always use threaded capture for this test
            capture_instances.append((name, camera))
            print(f"Connected to {name} camera")
        else:
            print(f"Failed to connect to {name} camera")

    try:
        while True:
            for name, camera in capture_instances:
                if camera.is_connected:
                    ret, frame = camera.read()
                    if ret:
                        cv2.imshow(f"{name}", frame)
                    else:
                        print(f"Failed to read frame from {name}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        for name, camera in capture_instances:
            if camera.is_connected:
                camera.stop()
            camera.disconnect()
            print(f"Disconnected from {name}")


# Example usage and testing
if __name__ == "__main__":
    # test_camera('basler')  # Change to 'ximea', 'ipcam', 'video_file', etc. as needed
    # test_camera('webcam', source=0)  # Change to 'ximea', 'ipcam', 'video_file', etc. as needed
    # test_camera('video_file', source="C:/Users/optane/Desktop/random_concat_20250604_112614.mp4", loop=True)  # Example for video file capture
    # test_camera('ximea')  # Change to 'ximea', 'ipcam', 'video_file', etc. as needed
    # test_camera('ipcam', source="rtsp://192.168.1.153:554/h264Preview_01_sub", username="admin", password="password")  # Example for IP camera capture
    # test_camera('folder', source="C:/Users/optane/Desktop/bird-calls-dataset/images/default", sort_by='date', fps=30, real_time=True, loop=True)

    cameras = [
        {'capture_type': 'basler', 'threaded': True},
        {'capture_type': 'ximea', 'threaded': True},
        {'capture_type': 'webcam', 'threaded': True},
        {'capture_type': 'ipcam', 'source': "rtsp://192.168.1.153:554/h264Preview_01_sub", 'username': "admin", 'password': "password", 'threaded': True},
        {'capture_type': 'video_file', 'source': "C:/Users/optane/Desktop/random_concat_20250604_112614.mp4", 'loop': True, 'threaded': True},
        {'capture_type': 'folder', 'source': "C:/Users/optane/Desktop/bird-calls-dataset/images/default", 'sort_by': 'date', 'fps': 30, 'real_time': True, 'loop': True, 'threaded': False}
    ]

    test_multiple_cameras(cameras, threaded=True)
