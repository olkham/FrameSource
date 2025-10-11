from typing import Optional, Tuple, Any, Dict
import numpy as np
import logging

try:
    from .video_capture_base import VideoCaptureBase
except ImportError:
    # If running as main script, try absolute import
    from video_capture_base import VideoCaptureBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaslerCapture(VideoCaptureBase):
    def start_async(self):
        """
        Start background thread to continuously capture frames from Basler camera.
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
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        frame = getattr(self, '_latest_frame', None)
        return (frame is not None), frame

    def _read_direct(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Directly read a frame from the Basler camera (bypassing background thread logic).
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self.is_connected or self.camera is None:
            return False, None
        try:
            grabResult = self.camera.RetrieveResult(5000, self.pylon.TimeoutHandling_ThrowException) # type: ignore
            if grabResult.GrabSucceeded():
                if self.converter:
                    image = self.converter.Convert(grabResult)
                    img_array = image.GetArray()
                else:
                    img_array = grabResult.Array
                grabResult.Release()
                return True, img_array
            else:
                grabResult.Release()
                return False, None
        except Exception as e:
            logger.error(f"Error reading from Basler camera: {e}")
            return False, None

    """Basler camera capture using pypylon."""
    
    def __init__(self, source: Any = None, **kwargs):
        super().__init__(source, **kwargs)
        self.camera = None
        self.converter = None
        try:
            from pypylon import pylon
            self.pylon = pylon
        except ImportError:
            logger.error("pypylon module not available. Install pypylon package.")
            self.pylon = None
        
        self.is_mono = kwargs.get('is_mono', False)
        self.serial_number = source if isinstance(source, str) else None
        self.device_index = source if isinstance(source, int) else 0
    
    def connect(self) -> bool:
        """Connect to Basler camera."""
        if self.pylon is None:
            logger.error("pypylon not available")
            return False
        
        try:
            # Get the transport layer factory
            tlFactory = self.pylon.TlFactory.GetInstance()
            
            # Get all attached devices and exit application if no device is found
            devices = tlFactory.EnumerateDevices()
            if len(devices) == 0:
                logger.error("No Basler cameras found")
                return False
            
            # Create camera object
            if self.serial_number:
                # Connect by serial number
                device_info = None
                for device in devices:
                    if device.GetSerialNumber() == self.serial_number:
                        device_info = device
                        break
                if device_info is None:
                    logger.error(f"Basler camera with serial number {self.serial_number} not found")
                    return False
                self.camera = self.pylon.InstantCamera(tlFactory.CreateDevice(device_info))
            else:
                # Connect by index
                if self.device_index >= len(devices):
                    logger.error(f"Basler camera index {self.device_index} out of range")
                    return False
                self.camera = self.pylon.InstantCamera(tlFactory.CreateDevice(devices[self.device_index]))
            
            # Open camera
            self.camera.Open()
            
            
            # Create image converter for color images
            self.converter = self.pylon.ImageFormatConverter()
            if self.is_mono:
                self.converter.OutputPixelFormat = self.pylon.PixelType_Mono8
            else:
                self.converter.OutputPixelFormat = self.pylon.PixelType_BGR8packed
            self.converter.OutputBitAlignment = self.pylon.OutputBitAlignment_MsbAligned
            
            # Apply config parameters
            if 'exposure' in self.config:
                self.set_exposure(self.config['exposure'])
            if 'gain' in self.config:
                self.set_gain(self.config['gain'])
            if 'width' in self.config and 'height' in self.config:
                self.set_frame_size(self.config['width'], self.config['height'])
            
            # Start grabbing
            self.camera.StartGrabbing(self.pylon.GrabStrategy_LatestImageOnly)
            
            self.is_connected = True
            logger.info(f"Connected to Basler camera {self.camera.GetDeviceInfo().GetModelName()}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Basler camera: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Basler camera."""
        try:
            if self.camera is not None:
                if self.camera.IsGrabbing():
                    self.camera.StopGrabbing()
                self.camera.Close()
                self.camera = None
            self.converter = None
            self.is_connected = False
            logger.info("Disconnected from Basler camera")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Basler camera: {e}")
            return False
    
    def _read_implementation(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Return the latest frame captured by the background thread, or fall back to direct read if not running.
        """
        if hasattr(self, '_capture_thread') and self._capture_thread is not None and self._capture_thread.is_alive():
            return self.get_latest_frame()
        else:
            return self._read_direct()
    
    def get_exposure_range(self) -> Tuple[float, float]:
        """Get exposure range in microseconds."""
        if not self.is_connected or self.camera is None:
            return (0.0, 0.0)
        
        try:
            min_exposure = self.camera.ExposureTime.GetMin()
            max_exposure = self.camera.ExposureTime.GetMax()
            return (min_exposure, max_exposure)
        except Exception as e:
            logger.error(f"Error getting exposure range: {e}")
            return (0.0, 0.0)
        
    def get_gain_range(self) -> Tuple[float, float]:
        """Get gain range in dB."""
        if not self.is_connected or self.camera is None:
            return (0.0, 0.0)
        
        try:
            min_gain = self.camera.Gain.GetMin()
            max_gain = self.camera.Gain.GetMax()
            return (min_gain, max_gain)
        except Exception as e:
            logger.error(f"Error getting gain range: {e}")
            return (0.0, 0.0)


    def set_exposure(self, value: float) -> bool:
        """Set exposure in microseconds."""
        if not self.is_connected or self.camera is None:
            return False
        
        try:
            self.camera.ExposureTime.SetValue(value)
            self._exposure = value
            return True
        except Exception as e:
            logger.error(f"Error setting exposure: {e}")
            return False
    
    def get_exposure(self) -> Optional[float]:
        """Get exposure in microseconds."""
        if not self.is_connected or self.camera is None:
            return self._exposure
        
        try:
            return self.camera.ExposureTime()
        except Exception:
            return self._exposure
    
    def set_gain(self, value: float) -> bool:
        """Set gain in dB."""
        if not self.is_connected or self.camera is None:
            return False
        try:
            self.camera.Gain.SetValue(value)
            self._gain = value
            return True
        except Exception as e:
            logger.error(f"Error setting gain: {e}")
            return False
    
    def get_gain(self) -> Optional[float]:
        """Get gain in dB."""
        if not self.is_connected or self.camera is None:
            return self._gain
        
        try:
            return self.camera.Gain.GetValue()
        except Exception:
            return self._gain
    
    def enable_auto_exposure(self, enable: bool = True) -> bool:
        """Enable or disable auto exposure for Basler camera."""
        if not self.is_connected or self.camera is None:
            return False
        try:
            if enable:
                self.camera.ExposureAuto.SetValue("Continuous")
            else:
                self.camera.ExposureAuto.SetValue("Off")
            logger.info(f"Set Basler auto exposure to {enable}")
            return True
        except Exception as e:
            logger.error(f"Error setting Basler auto exposure: {e}")
            return False
    
    def set_frame_size(self, width: int, height: int) -> bool:
        """Set frame size for Basler camera."""
        if not self.is_connected or self.camera is None:
            return False
        try:
            self.camera.Width.SetValue(width)
            self.camera.Height.SetValue(height)
            logger.info(f"Set Basler camera resolution to {width}x{height}")
            return True
        except Exception as e:
            logger.error(f"Error setting Basler camera resolution: {e}")
            return False
    
    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        """Get frame size."""
        if not self.is_connected or self.camera is None:
            return None
        
        try:
            width = self.camera.Width.GetValue()
            height = self.camera.Height.GetValue()
            return (width, height)
        except Exception:
            return None
    
    def set_fps(self, fps: float) -> bool:
        """Set FPS for Basler camera."""
        if not self.is_connected or self.camera is None:
            return False
        try:
            self.camera.AcquisitionFrameRateEnable.SetValue(True)
            self.camera.AcquisitionFrameRate.SetValue(fps)
            logger.info(f"Set Basler camera FPS to {fps}")
            return True
        except Exception as e:
            logger.error(f"Error setting Basler camera FPS: {e}")
            return False
    
    def get_fps(self) -> Optional[float]:
        """Get FPS."""
        if not self.is_connected or self.camera is None:
            return None
        
        try:
            return self.camera.AcquisitionFrameRate.GetValue()
        except Exception:
            return None

    @classmethod
    def discover(cls) -> list:
        """
        Discover available Basler cameras.
        
        Returns:
            list: List of dictionaries containing Basler camera information.
                Each dict contains: {'index': int, 'serial_number': str, 'name': str, 'device_class': str}
        """
        devices = []
        
        try:
            from pypylon import pylon
        except ImportError:
            logger.warning("pypylon module not available. Cannot discover Basler cameras.")
            return []
        
        try:
            # Get the transport layer factory
            tlFactory = pylon.TlFactory.GetInstance()
            
            # Get all attached devices
            device_list = tlFactory.EnumerateDevices()
            
            for index, device_info in enumerate(device_list):
                try:
                    device_data = {
                        'index': index,
                        'id': index,
                        'serial_number': device_info.GetSerialNumber(),
                        'name': device_info.GetFriendlyName(),
                        'device_class': device_info.GetDeviceClass()
                    }
                    devices.append(device_data)
                    logger.info(f"Found Basler camera: {device_data}")
                    
                except Exception as e:
                    logger.warning(f"Could not get info for Basler device {index}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error discovering Basler cameras: {e}")
        
        return devices

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Get configuration schema for Basler camera capture"""
        return {
            'title': 'Basler Camera Configuration',
            'description': 'Configure Basler industrial camera settings',
            'fields': [
                {
                    'name': 'source',
                    'label': 'Camera Source',
                    'type': 'text',
                    'placeholder': 'Serial number or device index (0, 1, 2...)',
                    'description': 'Camera serial number or device index',
                    'required': False,
                    'default': '0'
                },
                {
                    'name': 'exposure',
                    'label': 'Exposure Time (µs)',
                    'type': 'number',
                    'min': 1,
                    'max': 1000000,
                    'placeholder': '10000',
                    'description': 'Exposure time in microseconds',
                    'required': False
                },
                {
                    'name': 'gain',
                    'label': 'Gain (dB)',
                    'type': 'number',
                    'min': 0,
                    'max': 40,
                    'step': 0.1,
                    'placeholder': '0.0',
                    'description': 'Camera gain in decibels',
                    'required': False
                },
                {
                    'name': 'width',
                    'label': 'Width',
                    'type': 'number',
                    'min': 1,
                    'max': 10000,
                    'placeholder': '1920',
                    'description': 'Frame width in pixels',
                    'required': False
                },
                {
                    'name': 'height',
                    'label': 'Height',
                    'type': 'number',
                    'min': 1,
                    'max': 10000,
                    'placeholder': '1080',
                    'description': 'Frame height in pixels',
                    'required': False
                },
                {
                    'name': 'fps',
                    'label': 'Frame Rate (FPS)',
                    'type': 'number',
                    'min': 1,
                    'max': 1000,
                    'placeholder': '30',
                    'description': 'Frames per second',
                    'required': False,
                    'default': 30
                },
                {
                    'name': 'is_mono',
                    'label': 'Monochrome',
                    'type': 'checkbox',
                    'description': 'Camera outputs monochrome (grayscale) images',
                    'required': False,
                    'default': False
                }
            ]
        }


if __name__ == "__main__":
    # Example usage
    import cv2
    camera = BaslerCapture()  # Replace with actual serial number or index
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