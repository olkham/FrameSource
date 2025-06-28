from typing import Optional, Tuple
import numpy as np
import logging
from .video_capture_base import VideoCaptureBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XimeaCapture(VideoCaptureBase):
    def start(self):
        """
        Start background thread to continuously capture frames from Ximea camera.
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

    def get_latest_frame(self):
        """
        Get the most recent frame captured by the background thread.
        Returns:
            Optional[np.ndarray]: Latest frame or None if not available
        """
        return getattr(self, '_latest_frame', None)

    def _read_direct(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Directly read a frame from the Ximea camera (bypassing background thread logic).
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self.is_connected or self.cam is None:
            return False, None
        try:
            img = self.xiapi.Image()
            self.cam.get_image(img)
            data = img.get_image_data_numpy()
            return True, data
        except Exception as e:
            logger.error(f"Error reading from Ximea camera: {e}")
            return False, None

    """Ximea camera capture using xiapi."""
    
    def __init__(self, source: int = 0, **kwargs):
        super().__init__(source, **kwargs)
        self.cam = None
        try:
            from ximea import xiapi
            self.xiapi = xiapi
        except ImportError:
            logger.error("Ximea xiapi module not available. Install python-ximea package.")
            self.xiapi = None

        self.is_mono = kwargs.get('is_mono', False)

    def connect(self) -> bool:
        """Connect to Ximea camera."""
        if self.xiapi is None:
            logger.error("Ximea xiapi not available")
            return False
        
        try:
            self.cam = self.xiapi.Camera()
            # self.cam.open_device_by_SN(self.source) if isinstance(self.source, str) else self.cam.open_device(self.source)
            self.cam.open_device()
            # self.cam.set_imgdataformat('XI_RGB24')
            # Get the number of channels from the camera
            # try:
            #     channel_count = self.cam.get_param('imgdataformat')
            #     if channel_count == 'XI_MONO8':
            #         self.channel_count = 1
            #         self.is_mono = True
            #     elif channel_count == 'XI_RGB24':
            #         self.channel_count = 3
            #         self.is_mono = False
            #     else:
            #         # Fallback: use get_channel_count if available
            #         if hasattr(self.cam, 'get_channel_count'):
            #             self.channel_count = self.cam.get_channel_count()
            #         else:
            #             self.channel_count = None
            #     logger.info(f"Ximea camera channel count: {self.channel_count}")
            # except Exception as e:
            #     logger.warning(f"Could not determine channel count: {e}")
            #     self.channel_count = None

            # Set default parameters
            if not self.is_mono:
                self.cam.set_imgdataformat('XI_RGB24')
                self.cam.enable_auto_wb()
                # actual channel order = BGR
                # XI_RGB24 RGB data format. [Blue][Green][Red] (see Note5)
                # https://www.ximea.com/support/wiki/apis/xiapi_manual

            if self.is_mono:
                self.cam.set_imgdataformat('XI_MONO8')

            self.cam.set_exposure(10000)  # 10ms default
            
            # Apply config parameters
            if 'exposure' in self.config:
                self.cam.set_exposure(self.config['exposure'])
            if 'gain' in self.config:
                self.cam.set_gain(self.config['gain'])
            
            self.cam.start_acquisition()
            self.is_connected = True
            logger.info(f"Connected to Ximea camera {self.source}")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Ximea camera: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Ximea camera."""
        try:
            if self.cam is not None:
                self.cam.stop_acquisition()
                self.cam.close_device()
                self.cam = None
            self.is_connected = False
            logger.info("Disconnected from Ximea camera")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Ximea camera: {e}")
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
    
    def get_exposure_range(self) -> Optional[Tuple[float, float]]:
        """Get exposure range in microseconds."""
        if not self.is_connected or self.cam is None:
            return None, None
        
        try:
            min_exposure = self.cam.get_exposure_minimum()
            max_exposure = self.cam.get_exposure_maximum()
            return (min_exposure, max_exposure)
        except Exception as e:
            logger.error(f"Error getting exposure range: {e}")
            return None, None
        
    def get_gain_range(self) -> Optional[Tuple[float, float]]:
        """Get gain range in dB."""
        if not self.is_connected or self.cam is None:
            return None, None
        
        try:
            min_gain = self.cam.get_gain_minimum()
            max_gain = self.cam.get_gain_maximum()
            return (min_gain, max_gain)
        except Exception as e:
            logger.error(f"Error getting gain range: {e}")
            return None, None

    def set_exposure(self, value: float) -> bool:
        """Set exposure in microseconds."""
        if not self.is_connected or self.cam is None:
            return False
        
        try:
            self.cam.set_exposure(int(value))
            self._exposure = value
            return True
        except Exception as e:
            logger.error(f"Error setting exposure: {e}")
            return False
    
    def get_exposure(self) -> Optional[float]:
        """Get exposure in microseconds."""
        if not self.is_connected or self.cam is None:
            return self._exposure
        
        try:
            return float(self.cam.get_exposure())
        except Exception:
            return self._exposure
    
    def set_gain(self, value: float) -> bool:
        """Set gain in dB."""
        if not self.is_connected or self.cam is None:
            return False
        
        try:
            self.cam.set_gain(value)
            self._gain = value
            return True
        except Exception as e:
            logger.error(f"Error setting gain: {e}")
            return False
    
    def get_gain(self) -> Optional[float]:
        """Get gain in dB."""
        if not self.is_connected or self.cam is None:
            return self._gain
        
        try:
            return self.cam.get_gain()
        except Exception:
            return self._gain

    def enable_auto_exposure(self, enable: bool = True) -> bool:
        """
        Enable or disable auto exposure for Ximea camera.
        """
        if not self.is_connected or self.cam is None:
            return False
        try:
            if enable:
                self.cam.enable_aeag()
            else:
                self.cam.disable_aeag()
            logger.info(f"Set Ximea auto exposure to {enable}")
            return True
        except Exception as e:
            logger.error(f"Error setting Ximea auto exposure: {e}")
            return False
    
    def set_frame_size(self, width: int, height: int) -> bool:
        """Set frame size for Ximea camera."""
        if not self.is_connected or self.cam is None:
            return False
        try:
            self.cam.set_width(width)
            self.cam.set_height(height)
            logger.info(f"Set Ximea camera resolution to {width}x{height}")
            return True
        except Exception as e:
            logger.error(f"Error setting Ximea camera resolution: {e}")
            return False

if __name__ == "__main__":
    # Example usage
    import cv2
    camera = XimeaCapture(is_mono=False)
    if camera.connect():
        camera.start()
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

        camera.stop()
        camera.disconnect()
    else:
        print("Failed to connect to webcam.")