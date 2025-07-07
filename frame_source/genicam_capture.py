from typing import Optional, Tuple, Any
import numpy as np
import logging
from .video_capture_base import VideoCaptureBase
from genicam.gentl import TimeoutException

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GenicamCapture(VideoCaptureBase):
    def start(self):
        """
        Start background thread to continuously capture frames from a Genicam compliant camera.
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

    @staticmethod
    def to_np(buffer):
        from harvesters.util.pfnc import mono_location_formats, \
            rgb_formats, bgr_formats, \
            rgba_formats, bgra_formats, bayer_location_formats

        payload = buffer.payload
        component = payload.components[0]
        width = component.width
        height = component.height
        data_format = component.data_format

        # Reshape the image so that it can be drawn on the VisPy canvas:
        if data_format in mono_location_formats:
            content = component.data.reshape(height, width)
        else:
            # The image requires you to reshape it to draw it on the
            # canvas:
            if data_format in rgb_formats or \
                    data_format in rgba_formats or \
                    data_format in bgr_formats or \
                    data_format in bgra_formats or \
                    data_format in bayer_location_formats:
                #
                content = np.copy(component.data.reshape(
                    height, width,
                    int(component.num_components_per_pixel)  # Set of R, G, B, and Alpha
                ))
                #
                if data_format in rgb_formats:
                    # Swap every R and B:
                    content = content[:, :, ::-1]
            else:
                ycbcr422_data = np.copy(component.data.reshape((-1, 4)))

                Y0 = ycbcr422_data[:, 0].astype(np.float32)
                Cb = ycbcr422_data[:, 1].astype(np.float32)
                Y1 = ycbcr422_data[:, 2].astype(np.float32)
                Cr = ycbcr422_data[:, 3].astype(np.float32)

                # Expand to per-pixel arrays
                Y = np.empty((ycbcr422_data.shape[0] * 2,), dtype=np.float32)
                Cb_full = np.empty_like(Y)
                Cr_full = np.empty_like(Y)

                Y[0::2] = Y0
                Y[1::2] = Y1
                Cb_full[0::2] = Cb
                Cb_full[1::2] = Cb
                Cr_full[0::2] = Cr
                Cr_full[1::2] = Cr
                # Limited range conversion (BT.601)
                # Scale Y component
                Y = Y.astype(np.float64) - 16
                Y = np.clip(Y, 0, 255)

                # Constants for limited-range YCbCr
                R = 1.164 * Y + 1.596 * (Cr_full - 128)
                G = 1.164 * Y - 0.392 * (Cb_full - 128) - 0.813 * (Cr_full - 128)
                B = 1.164 * Y + 2.017 * (Cb_full - 128)

                # Clip values to valid range [0, 255] and convert to uint8
                R = np.clip(R, 0, 255).astype(np.uint8)
                G = np.clip(G, 0, 255).astype(np.uint8)
                B = np.clip(B, 0, 255).astype(np.uint8)
                # Stack into final image
                rgb_image = np.stack((B, G, R), axis=-1).reshape((height, width, 3))
                return rgb_image
        return content

    def release_buffers(self):
        for _buffer in self._buffers:
            if _buffer:
                _buffer.queue()
        self._buffers.clear()

    def _read_direct(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Directly read a frame from the Genicam compliant camera (bypassing background thread logic).
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self.is_connected or self.camera is None:
            return False, None
        try:
            buffer = self.camera.fetch(timeout=0.0001)
            # self._prepare_texture(buffer)


            nump = self.to_np(buffer)

            self.release_buffers()
            self._buffers.append(buffer)

            return True, nump
        except TimeoutException:
            # We have an ImageAcquirer object but nothing has
            # been fetched, wait for the next round:
            return False, None
        except Exception as e:
            logger.error(f"Error reading from Genicam camera: {e}")
            raise e
            return False, None

    """Genicam camera capture using pypylon."""
    
    def __init__(self, source: Any = None, **kwargs):
        super().__init__(source, **kwargs)
        self.camera = None
        self.converter = None
        try:
            from harvesters.core import Harvester
            self.h = Harvester()
        except ImportError:
            logger.error("Harvesters module not available. Install Harvesters package.")
            self.h = None

        self.is_mono = kwargs.get('is_mono', False)
        self.serial_number = source if isinstance(source, str) else None
        self.device_index = source if isinstance(source, int) else 0
        self._buffers = []

    def connect(self) -> bool:
        """Connect to Genicam camera."""
        if self.h is None:
            logger.error("Harvesters not available")
            return False
        
        try:
            self.h.add_file('/opt/pylon/lib/gentlproducer/gtl/ProducerU3V.cti')
            # h.add_file('/opt/pylon/lib/gentlproducer/gtl/ProducerGEV.cti')
            # h.add_file('/usr/lib/ids/cti/ids_gevgentl.cti')
            # h.add_file('/usr/lib/ids/cti/ids_u3vgentl.cti')
            self.h.add_file('/usr/lib/ids/cti/ids_ueyegentl.cti')
            self.h.update()
            self._buffers = []

            devices = self.h.device_info_list

            if len(devices) == 0:
                logger.error("No Genicam cameras found")
                return False
            
            # Create camera object
            if self.serial_number:
                self.camera = self.h.create({'serial_number': self.serial_number})
            else:
                self.camera = self.h.create(self.device_index)
            
            # Open camera
            n = self.camera.remote_device.node_map
            # Change camera properties to listen for Bruker TTL triggers
            # n.TriggerSelector.value = "SingleFrameTrigger" <-- currently not changeable...
            n.TriggerMode.value = "Off"
            n.TriggerActivation.value = "RisingEdge"
            n.TriggerSource.value = "Line2"
            n.LineSelector.value = "Line2"


            self.camera.start()
            
            # # Create image converter for color images
            # self.converter = self.pylon.ImageFormatConverter()
            # if self.is_mono:
            #     self.converter.OutputPixelFormat = self.pylon.PixelType_Mono8
            # else:
            #     self.converter.OutputPixelFormat = self.pylon.PixelType_BGR8packed
            # self.converter.OutputBitAlignment = self.pylon.OutputBitAlignment_MsbAligned
            
            # Apply config parameters
            if 'exposure' in self.config:
                self.set_exposure(self.config['exposure'])
            if 'gain' in self.config:
                self.set_gain(self.config['gain'])
            if 'width' in self.config and 'height' in self.config:
                self.set_frame_size(self.config['width'], self.config['height'])
            

            self.is_connected = True

            logger.info(f"Connected to Genicam camera {self.camera.device.module.model}")
            return True
            
        except Exception as e:
            logger.error(f"Error connecting to Genicam camera: {e}")
            return False
    
    def disconnect(self) -> bool:
        """Disconnect from Genicam camera."""
        try:
            if self.h is not None:
                self.h.reset()
            logger.info("Disconnected from Genicam camera")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Genicam camera: {e}")
            return False
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
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
        """Enable or disable auto exposure for Genicam camera."""
        if not self.is_connected or self.camera is None:
            return False
        try:
            if enable:
                self.camera.ExposureAuto.SetValue("Continuous")
            else:
                self.camera.ExposureAuto.SetValue("Off")
            logger.info(f"Set Genicam auto exposure to {enable}")
            return True
        except Exception as e:
            logger.error(f"Error setting Genicam auto exposure: {e}")
            return False
    
    def set_frame_size(self, width: int, height: int) -> bool:
        """Set frame size for Genicam camera."""
        if not self.is_connected or self.camera is None:
            return False
        try:
            self.camera.Width.SetValue(width)
            self.camera.Height.SetValue(height)
            logger.info(f"Set Genicam camera resolution to {width}x{height}")
            return True
        except Exception as e:
            logger.error(f"Error setting Genicam camera resolution: {e}")
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
        """Set FPS for Genicam camera."""
        if not self.is_connected or self.camera is None:
            return False
        try:
            self.camera.AcquisitionFrameRateEnable.SetValue(True)
            self.camera.AcquisitionFrameRate.SetValue(fps)
            logger.info(f"Set Genicam camera FPS to {fps}")
            return True
        except Exception as e:
            logger.error(f"Error setting Genicam camera FPS: {e}")
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
    camera = GenicamCapture()  # Replace with actual serial number or index
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