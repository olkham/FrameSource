from typing import Optional, Tuple, Any
import numpy as np
import logging
import platform
from video_capture_base import VideoCaptureBase
import mvsdk as mvsdk

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class HuatengCapture(VideoCaptureBase):
    """Huateng camera capture using mvsdk."""
    
    def __init__(self, source: Any = None, **kwargs):
        super().__init__(source, **kwargs)
        self.hCamera = -1
        self.nDev = 0
        self.capability = None
        self.pFrameBuffer = None
        self.frame = None
        self.DevInfo = None
        self.is_mono = kwargs.get('is_mono', False)
        self.current_exp = 0
        self.current_gain = 0
        self.prop_frame_height = None
        self.prop_frame_width = None
        self.is_new_frame = False
        self.read_count = 0

    def connect(self) -> bool:
        try:
            self.DevList = mvsdk.CameraEnumerateDevice()
            self.nDev = len(self.DevList)
            if self.nDev < 1:
                logger.error("No Huateng cameras found!")
                return False
            i = 0  # Always pick the first camera for now
            self.DevInfo = self.DevList[i]
            self.hCamera = mvsdk.CameraInit(self.DevInfo, -1, -1)
            self.capability = mvsdk.CameraGetCapability(self.hCamera)
            monoCamera = (self.capability.sIspCapacity.bMonoSensor != 0)
            if monoCamera:
                mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
            else:
                mvsdk.CameraSetIspOutFormat(self.hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
            mvsdk.CameraSetTriggerMode(self.hCamera, 0)
            mvsdk.CameraSetAeState(self.hCamera, 1)
            mvsdk.CameraPlay(self.hCamera)
            FrameBufferSize = self.capability.sResolutionRange.iWidthMax * self.capability.sResolutionRange.iHeightMax * (1 if monoCamera else 3)
            self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
            self.prop_frame_height = self.capability.sResolutionRange.iHeightMax
            self.prop_frame_width = self.capability.sResolutionRange.iWidthMax
            self.is_connected = True
            logger.info("Connected to Huateng camera")
            return True
        except Exception as e:
            logger.error(f"Error connecting to Huateng camera: {e}")
            return False

    def disconnect(self) -> bool:
        try:
            if self.hCamera != -1:
                mvsdk.CameraUnInit(self.hCamera)
                self.hCamera = -1
            if self.pFrameBuffer is not None:
                mvsdk.CameraAlignFree(self.pFrameBuffer)
                self.pFrameBuffer = None
            self.is_connected = False
            logger.info("Disconnected from Huateng camera")
            return True
        except Exception as e:
            logger.error(f"Error disconnecting from Huateng camera: {e}")
            return False

    def _read_direct(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_connected or self.hCamera == -1:
            return False, None
        try:
            pRawData, FrameHead = mvsdk.CameraGetImageBuffer(self.hCamera, 10)
            mvsdk.CameraImageProcess(self.hCamera, pRawData, self.pFrameBuffer, FrameHead)
            mvsdk.CameraReleaseImageBuffer(self.hCamera, pRawData)
            if platform.system() == "Windows":
                mvsdk.CameraFlipFrameBuffer(self.pFrameBuffer, FrameHead, 1)
            frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(self.pFrameBuffer)
            frame = np.frombuffer(frame_data, dtype=np.uint8)
            frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3))
            return True, frame
        except Exception as e:
            logger.error(f"Error reading from Huateng camera: {e}")
            return False, None

    def _read_implementation(self) -> Tuple[bool, Optional[np.ndarray]]:
        if hasattr(self, '_capture_thread') and self._capture_thread is not None and self._capture_thread.is_alive():
            return self.get_latest_frame()
        else:
            return self._read_direct()

    def set_exposure(self, value: float) -> bool:
        try:
            mvsdk.CameraSetAeState(self.hCamera, False)
            mvsdk.CameraSetExposureTime(self.hCamera, int(value))
            self.current_exp = int(value)
            return True
        except Exception as e:
            logger.error(f"Error setting exposure: {e}")
            return False

    def get_exposure(self) -> Optional[float]:
        try:
            return float(mvsdk.CameraGetExposureTime(self.hCamera))
        except Exception:
            return self.current_exp

    def set_gain(self, value: float) -> bool:
        try:
            mvsdk.CameraSetAeState(self.hCamera, False)
            mvsdk.CameraSetAnalogGainX(self.hCamera, int(value))
            self.current_gain = int(value)
            return True
        except Exception as e:
            logger.error(f"Error setting gain: {e}")
            return False

    def get_gain(self) -> Optional[float]:
        try:
            return float(mvsdk.CameraGetAnalogGainX(self.hCamera))
        except Exception:
            return self.current_gain

    def enable_auto_exposure(self, enable: bool = True) -> bool:
        try:
            mvsdk.CameraSetAeState(self.hCamera, bool(enable))
            logger.info(f"Set Huateng auto exposure to {enable}")
            return True
        except Exception as e:
            logger.error(f"Error setting auto exposure: {e}")
            return False

    def get_exposure_range(self) -> Optional[Tuple[float, float]]:
        try:
            min_exp = 5
            max_exp = 31 * 1000
            return (min_exp, max_exp)
        except Exception:
            return None

    def get_gain_range(self) -> Optional[Tuple[float, float]]:
        try:
            min_gain = 2
            max_gain = 6
            return (min_gain, max_gain)
        except Exception:
            return None

    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        if self.prop_frame_width and self.prop_frame_height:
            return (self.prop_frame_width, self.prop_frame_height)
        return None

    def set_frame_size(self, width: int, height: int) -> bool:
        try:
            imageres = mvsdk.tSdkImageResolution(width, height)
            mvsdk.CameraSetImageResolution(self.hCamera, imageres)
            self.prop_frame_width = width
            self.prop_frame_height = height
            logger.info(f"Set Huateng camera resolution to {width}x{height}")
            return True
        except Exception as e:
            logger.error(f"Error setting Huateng camera resolution: {e}")
            return False

    def get_fps(self) -> Optional[float]:
        # Not directly supported in mvsdk, return None
        return None

    def set_fps(self, fps: float) -> bool:
        # Not directly supported in mvsdk, return False
        return False

if __name__ == "__main__":
    # Example usage
    import cv2
    camera = HuatengCapture(is_mono=False)
    if camera.connect():
        camera.start()
        print("Webcam connected successfully.")
        print(f"Exposure: {camera.get_exposure()}")
        print(f"Gain: {camera.get_gain()}")
        print(f"Frame size: {camera.get_frame_size()}")
        
        # Read a few frames
        while camera.is_connected:
            ret, frame = camera.read()
            if ret and frame is not None:
                cv2.imshow("camera", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        camera.stop()
        camera.disconnect()
    else:
        print("Failed to connect to webcam.")