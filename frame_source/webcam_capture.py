from typing import Optional, Tuple, Any, Dict
import numpy as np
import cv2
import logging
import platform
import sys

try:
    from .video_capture_base import VideoCaptureBase
except ImportError:
    # If running as main script, try absolute import
    from video_capture_base import VideoCaptureBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebcamCapture(VideoCaptureBase):
    def start_async(self):
        """
        Start background thread to continuously capture frames from webcam.
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
            # time.sleep(0.01)  # ~100 FPS max, adjust as needed

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the most recent frame captured by the background thread.
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        frame = getattr(self, '_latest_frame', None)
        self._latest_frame = None  # Clear after reading to avoid stale data
        return (frame is not None), frame

    def _read_direct(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Directly read a frame from the webcam (bypassing background thread logic).
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self.is_connected or self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        return ret, frame if ret else None

    """Webcam capture using OpenCV."""
    
    def __init__(self, source: int = 0, **kwargs):
        super().__init__(source, **kwargs)
        self.cap = None
        # Set API preference based on OS
        if platform.system() == "Windows":
            self.api_preference = cv2.CAP_DSHOW  # DirectShow for Windows
        elif platform.system() == "Darwin":
            self.api_preference = cv2.CAP_AVFOUNDATION  # AVFoundation for macOS
        else:
            self.api_preference = cv2.CAP_V4L2   # Video4Linux for Linux

        self.source = source

        if 'is_mono' in kwargs:
            logger.warning("'is_mono' argument is only used for certain industrial cameras and has no effect for webcams.")
        
    def connect(self) -> bool:
        """Connect to webcam."""
        try:
            src = self.source
            api_pref = self.api_preference
            # Support `api_pref/index` format or `api_pref/path` format used in list_devices `id` field
            if isinstance(src, str) and ":" in src:
                parts = src.split(":")
                api_pref, src = parts[0], parts[1]

                if api_pref.isdigit():
                    api_pref = int(api_pref)

                if src.isdigit():
                    src = int(src)

            self.cap = cv2.VideoCapture(src, api_pref)
            if not self.cap.isOpened():
                logger.error(f"Failed to open webcam {src}")
                return False
            
            # Set additional parameters if provided
            if 'width' in self.config and 'height' in self.config:
                self.set_frame_size(self.config['width'], self.config['height'])
            if 'fps' in self.config:
                self.cap.set(cv2.CAP_PROP_FPS, self.config['fps'])
                
            self.is_connected = True
            logger.info(f"Connected to webcam {src}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to webcam: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from webcam."""
        try:
            if self.cap is not None:
                self.cap.release()
                self.cap = None
            self.is_connected = False
            logger.info("Disconnected from webcam")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from webcam: {e}")
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
        """Set exposure (-13 to -1 for most webcams)."""
        if not self.is_connected or self.cap is None:
            return False
        
        try:
            self.cap.set(cv2.CAP_PROP_EXPOSURE, value)
            self._exposure = value
            return True
        except Exception as e:
            logger.error(f"Error setting exposure: {e}")
            return False
    
    def get_exposure(self) -> Optional[float]:
        """Get current exposure."""
        if not self.is_connected or self.cap is None:
            return None
        
        try:
            return self.cap.get(cv2.CAP_PROP_EXPOSURE)
        except Exception:
            return self._exposure
    
    def set_gain(self, value: float) -> bool:
        """Set gain (0-255 for most webcams)."""
        if not self.is_connected or self.cap is None:
            return False
        
        try:
            self.cap.set(cv2.CAP_PROP_GAIN, value)
            self._gain = value
            return True
        except Exception as e:
            logger.error(f"Error setting gain: {e}")
            return False
    
    def get_gain(self) -> Optional[float]:
        """Get current gain."""
        if not self.is_connected or self.cap is None:
            return None
        
        try:
            return self.cap.get(cv2.CAP_PROP_GAIN)
        except Exception:
            return self._gain
    
    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        """Get frame size."""
        if not self.is_connected or self.cap is None:
            return None
        
        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return (width, height)
    
    def set_frame_size(self, width: int, height: int) -> bool:
        """Set frame size."""
        if self.cap is None:
            return False
        result1 = self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        result2 = self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        logger.info(f"Set webcam resolution to {width}x{height} (success: {result1 and result2})")
        return result1 and result2
    
    def get_fps(self) -> Optional[float]:
        """Get FPS."""
        if not self.is_connected or self.cap is None:
            return None
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def set_fps(self, fps: float) -> bool:
        """Set FPS."""
        if not self.is_connected or self.cap is None:
            return False
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        return True

    def enable_auto_exposure(self, enable: bool = True) -> bool:
        """
        Enable or disable auto exposure for webcam.
        """
        if not self.is_connected or self.cap is None:
            return False
        try:
            # OpenCV expects 0.75 for auto, 0.25 for manual (on many webcams)
            value = 0.75 if enable else 0.25
            result = self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, value)
            logger.info(f"Set auto exposure to {enable} (cv2 value: {value})")
            return result
        except Exception as e:
            logger.error(f"Error setting auto exposure: {e}")
            return False

    @classmethod
    def discover(cls) -> list:
        try:
            devices = []
            from cv2_enumerate_cameras import enumerate_cameras
            from cv2.videoio_registry import getBackendName

            camera_list = []
            if platform.system() == "Windows":
                backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
            elif platform.system() == "Darwin":  # macOS
                backends = [cv2.CAP_AVFOUNDATION]
            else:  # Linux
                backends = [cv2.CAP_V4L2]

            for backend in backends:
                camera_list.extend(enumerate_cameras(backend))

            for camera_info in camera_list:
                devices.append({"id": f"{camera_info.backend}:{camera_info.index}:{camera_info.path}","index":camera_info.index, "name":camera_info.name, "backend_index": camera_info.backend, "backend_name":getBackendName(camera_info.backend)})
            logger.info(f"Found {cls.__name__} input device: {devices}")
            return devices
        except ImportError:
            logger.warning("cv2-enumerate-cameras module not available. Install cv2-enumerate-cameras to list available (web)cameras.")
        return []

    @classmethod
    def _discover(cls) -> list:
        """
        Discover available webcam devices with real device names.
        
        Returns:
            list: List of dictionaries containing webcam device information.
                Each dict contains: {'index': int, 'name': str}
        """
        devices = []
        
        # Platform-specific device enumeration
        if platform.system() == "Windows":
            devices = cls._discover_windows_cameras()
        elif platform.system() == "Darwin":  # macOS
            devices = cls._discover_macos_cameras()
        else:  # Linux
            devices = cls._discover_linux_cameras()
        
        return devices
    
    @classmethod
    def _discover_windows_cameras(cls) -> list:
        """Discover cameras on Windows with proper device names."""
        devices = []
        
        # First, get all camera names from Windows
        camera_names = cls._get_all_windows_camera_names()
        
        # Then verify which indices actually work with OpenCV
        for i, name in enumerate(camera_names):
            # Try to open with DirectShow (most reliable for device enumeration)
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                # Try to read a frame to verify it actually works
                ret, _ = cap.read()
                if ret:
                    devices.append({
                        'index': i,
                        'name': name
                    })
                    logger.info(f"Found webcam: Index {i}, Name: {name}")
                cap.release()
            else:
                # Camera name exists in system but can't be opened
                # Still add it but mark as potentially unavailable
                devices.append({
                    'index': i,
                    'name': name + " (unavailable)"
                })
        
        return devices
    
    @classmethod
    def _get_all_windows_camera_names(cls) -> list:
        """Get all camera device names from Windows."""
        camera_names = []
        
        try:
            import subprocess
            
            # Use PowerShell to get camera devices
            # This queries the actual device manager entries
            ps_cmd = """
Get-CimInstance Win32_PnPEntity | Where-Object {
    $_.PNPClass -eq 'Camera' -or $_.PNPClass -eq 'Image'
} | Where-Object {
    $_.Status -eq 'OK'
} | Select-Object -ExpandProperty Caption | Sort-Object
"""
            
            result = subprocess.run(
                ['powershell', '-NoProfile', '-Command', ps_cmd],
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
            )
            
            if result.returncode == 0 and result.stdout.strip():
                camera_names = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
                
        except Exception as e:
            logger.warning(f"Could not enumerate Windows cameras via PowerShell: {e}")
            
            # Fallback: Try alternate method using WMI query
            try:
                ps_cmd_alt = "Get-WmiObject Win32_PnPEntity | Where-Object {$_.Name -match 'camera|webcam'} | Select-Object -ExpandProperty Name"
                result = subprocess.run(
                    ['powershell', '-NoProfile', '-Command', ps_cmd_alt],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, 'CREATE_NO_WINDOW') else 0
                )
                
                if result.returncode == 0 and result.stdout.strip():
                    camera_names = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
            except:
                pass
        
        return camera_names
    
    @classmethod
    def _discover_macos_cameras(cls) -> list:
        """Discover cameras on macOS."""
        devices = []
        
        try:
            import subprocess
            import json
            
            # Use system_profiler to get camera information
            cmd = ['system_profiler', 'SPCameraDataType', '-json']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                data = json.loads(result.stdout)
                cameras = data.get('SPCameraDataType', [])
                
                for i, camera in enumerate(cameras):
                    camera_name = camera.get('_name', f'Camera {i}')
                    
                    # Verify it works with OpenCV
                    cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        if ret:
                            devices.append({
                                'index': i,
                                'name': camera_name
                            })
                            logger.info(f"Found webcam: Index {i}, Name: {camera_name}")
                        cap.release()
                        
        except Exception as e:
            logger.warning(f"Could not enumerate macOS cameras: {e}")
            # Fallback to index-based detection
            for i in range(10):
                cap = cv2.VideoCapture(i, cv2.CAP_AVFOUNDATION)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        devices.append({
                            'index': i,
                            'name': f'Camera {i}'
                        })
                    cap.release()
        
        return devices
    
    @classmethod
    def _discover_linux_cameras(cls) -> list:
        """Discover cameras on Linux."""
        devices = []
        
        try:
            import os
            import glob
            
            # Look for video devices in /sys/class/video4linux/
            video_devices = glob.glob('/sys/class/video4linux/video*')
            video_devices.sort()
            
            for device_path in video_devices:
                try:
                    # Extract device number (e.g., video0 -> 0)
                    device_name = os.path.basename(device_path)
                    if device_name.startswith('video'):
                        device_num = int(device_name[5:])
                        
                        # Try to read the device name
                        name_file = os.path.join(device_path, 'name')
                        camera_name = f'Camera {device_num}'
                        
                        if os.path.exists(name_file):
                            with open(name_file, 'r') as f:
                                name_from_file = f.read().strip()
                                if name_from_file:
                                    camera_name = name_from_file
                        
                        # Verify it works with OpenCV
                        cap = cv2.VideoCapture(device_num, cv2.CAP_V4L2)
                        if cap.isOpened():
                            ret, _ = cap.read()
                            if ret:
                                devices.append({
                                    'index': device_num,
                                    'name': camera_name
                                })
                                logger.info(f"Found webcam: Index {device_num}, Name: {camera_name}")
                            cap.release()
                            
                except (ValueError, IOError, OSError) as e:
                    logger.debug(f"Error checking device {device_path}: {e}")
                    continue
                    
        except Exception as e:
            logger.warning(f"Could not enumerate Linux cameras: {e}")
        
        return devices

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Get configuration schema for webcam capture"""
        return {
            'title': 'Webcam Configuration',
            'description': 'Configure USB webcam or built-in camera settings',
            'fields': [
                {
                    'name': 'source',
                    'label': 'Camera Index',
                    'type': 'number',
                    'min': 0,
                    'max': 10,
                    'placeholder': '0',
                    'description': 'Camera device index (0 for default, 1 for second camera, etc.)',
                    'required': False,
                    'default': 0
                },
                {
                    'name': 'width',
                    'label': 'Width',
                    'type': 'number',
                    'min': 160,
                    'max': 4096,
                    'placeholder': '1920',
                    'description': 'Frame width in pixels',
                    'required': False
                },
                {
                    'name': 'height',
                    'label': 'Height',
                    'type': 'number',
                    'min': 120,
                    'max': 2160,
                    'placeholder': '1080',
                    'description': 'Frame height in pixels',
                    'required': False
                },
                {
                    'name': 'fps',
                    'label': 'Frame Rate (FPS)',
                    'type': 'number',
                    'min': 1,
                    'max': 120,
                    'placeholder': '30',
                    'description': 'Frames per second',
                    'required': False,
                    'default': 30
                },
                {
                    'name': 'exposure',
                    'label': 'Exposure',
                    'type': 'number',
                    'min': -13,
                    'max': -1,
                    'step': 1,
                    'placeholder': '-6',
                    'description': 'Manual exposure value (-13 to -1, lower = brighter)',
                    'required': False
                },
                {
                    'name': 'gain',
                    'label': 'Gain',
                    'type': 'number',
                    'min': 0,
                    'max': 255,
                    'placeholder': '0',
                    'description': 'Camera gain (0-255)',
                    'required': False
                }
            ]
        }


if __name__ == "__main__":
    # Example usage
    
    print("Discovering webcams...")
    webcams = WebcamCapture.discover()
    print(f"\nFound {len(webcams)} webcam(s):\n")
    for cam in webcams:
        print(f"  [{cam['index']}] {cam['name']}")
    
    if webcams:
        print(f"\nTesting first camera...")
        camera = WebcamCapture(source=0)
        if camera.connect():
            print("Webcam connected successfully.")
            print(f"Exposure: {camera.get_exposure()}")
            print(f"Gain: {camera.get_gain()}")
            print(f"Frame size: {camera.get_frame_size()}")
            
            # Read a few frames
            print("Press 'q' to quit...")
            while camera.is_connected:
                ret, frame = camera.read()
                if ret:
                    cv2.imshow("Webcam", frame) # type: ignore
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                else:
                    print("Failed to read frame from webcam.")
                    break

            camera.disconnect()
        else:
            print("Failed to connect to webcam.")
    else:
        print("\nNo webcams found.")