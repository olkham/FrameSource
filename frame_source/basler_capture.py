from typing import Optional, Tuple, Any
import numpy as np
import logging
from .video_capture_base import VideoCaptureBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaslerCapture(VideoCaptureBase):
    def start(self):
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
        while not self._stop_event.is_set():
            success, frame = self._read_direct()
            if success:
                self._latest_frame = frame
            time.sleep(0.01)  # ~100 FPS max, adjust as needed

    def get_latest_frame(self):
        """
        Get the most recent frame captured by the background thread.
        Returns:
            Optional[np.ndarray]: Latest frame or None if not available
        """
        return getattr(self, '_latest_frame', None)

    def _read_direct(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Directly read a frame from the Basler camera (bypassing background thread logic).
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self.is_connected or self.camera is None:
            return False, None
        try:
            grabResult = self.camera.RetrieveResult(5000, self.pylon.TimeoutHandling_ThrowException)
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
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Return the latest frame captured by the background thread, or fall back to direct read if not running.
        """
        if hasattr(self, '_capture_thread') and self._capture_thread is not None and self._capture_thread.is_alive():
            frame = self.get_latest_frame()
            return (frame is not None), frame
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
            return self.camera.ExposureTime.value
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
            if ret:
                cv2.imshow("Webcam", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        camera.disconnect()
    else:
        print("Failed to connect to webcam.")