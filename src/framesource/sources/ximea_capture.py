from typing import Optional, Tuple
import numpy as np
import logging
from .video_capture_base import VideoCaptureBase
from ..errors import MissingDependencyError

logger = logging.getLogger(__name__)

class XimeaCapture(VideoCaptureBase):
    supports_exposure = True
    supports_gain = True

    """Ximea camera capture using xiapi."""

    def __init__(self, source: int = 0, *, is_mono: Optional[bool] = None,
                 exposure: Optional[float] = None, gain: Optional[float] = None,
                 **kwargs):
        """Initialize the Ximea capture.

        Args:
            source: Camera device index (default: 0).
            is_mono: Configure the camera for monochrome (``XI_MONO8``)
                output instead of ``XI_RGB24`` (default: False). Applied on
                :meth:`connect`.
            exposure: Initial exposure time in microseconds. Applied on
                :meth:`connect` when provided (overrides the 10ms default).
            gain: Initial gain in dB. Applied on :meth:`connect` when
                provided.
            **kwargs: Additional passthrough options stored on ``self.config``.

        Raises:
            MissingDependencyError: If the ``ximea`` (xiapi) package is not
                installed.
        """
        # Preserve the historical ``self.config`` contents: only forward
        # explicitly-provided values so the ``'exposure' in self.config``
        # style presence checks in connect() keep their old meaning.
        for _key, _val in (
            ('is_mono', is_mono), ('exposure', exposure), ('gain', gain),
        ):
            if _val is not None:
                kwargs[_key] = _val
        super().__init__(source, **kwargs)
        self.cam = None
        try:
            from ximea import xiapi
            self.xiapi = xiapi
        except ImportError as e:
            raise MissingDependencyError(
                'ximea',
                extra=None,
                details=(
                    f"{e}; no pip extra is available for the Ximea xiapi bindings. "
                    "Download and install the XIMEA Software Package for your OS from "
                    "https://www.ximea.com/support/wiki/apis/xiapi_manual, which provides "
                    "the 'ximea' Python module."
                ),
            ) from e

        self.is_mono = self.config.get('is_mono', False)

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
    
    def _read_implementation(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the Ximea camera.
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self.is_connected or self.cam is None or self.xiapi is None:
            return False, None
        try:
            img = self.xiapi.Image()
            self.cam.get_image(img)
            data = img.get_image_data_numpy()
            return True, data
        except Exception as e:
            logger.error(f"Error reading from Ximea camera: {e}")
            return False, None
    
    def get_exposure_range(self) -> Optional[Tuple[float, float]]:
        """Get exposure range in microseconds."""
        if not self.is_connected or self.cam is None:
            return None
        
        try:
            min_exposure = self.cam.get_exposure_minimum()
            max_exposure = self.cam.get_exposure_maximum()
            if min_exposure is None or max_exposure is None:
                return None
            return (float(min_exposure), float(max_exposure))
        except Exception as e:
            logger.error(f"Error getting exposure range: {e}")
            return None
        
    def get_gain_range(self) -> Optional[Tuple[float, float]]:
        """Get gain range in dB."""
        if not self.is_connected or self.cam is None:
            return None
        
        try:
            min_gain = self.cam.get_gain_minimum()
            max_gain = self.cam.get_gain_maximum()
            if min_gain is None or max_gain is None:
                return None
            return (float(min_gain), float(max_gain))
        except Exception as e:
            logger.error(f"Error getting gain range: {e}")
            return None

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
            exposure_value = self.cam.get_exposure()
            if exposure_value is not None:
                return float(exposure_value)
            return self._exposure
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
            gain_value = self.cam.get_gain()
            if gain_value is not None:
                return float(gain_value)
            return self._gain
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
        print("Webcam connected successfully.")
        print(f"Exposure: {camera.get_exposure()}")
        print(f"Gain: {camera.get_gain()}")
        print(f"Frame size: {camera.get_frame_size()}")
        
        # Read a few frames
        while camera.is_connected:
            ret, frame = camera.read()
            if ret or frame is not None:
                cv2.imshow("Webcam", frame) # type: ignore
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        camera.stop()
        camera.disconnect()
    else:
        print("Failed to connect to webcam.")