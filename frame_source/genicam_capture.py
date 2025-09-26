import logging
from typing import Optional, Tuple, Any, Dict

import numpy as np

try:
    from .video_capture_base import VideoCaptureBase
except ImportError:
    from video_capture_base import VideoCaptureBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenicamCapture(VideoCaptureBase):
    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Get configuration schema for GenICam capture"""
        return {
            'title': 'GenICam Camera Configuration',
            'description': 'Configure GenICam compliant camera settings',
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
                    'name': 'cti_files',
                    'label': 'CTI Files',
                    'type': 'text',
                    'placeholder': '/path/to/producer.cti',
                    'description': 'GenTL producer files (comma-separated)',
                    'required': False
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
                    'max': 240,
                    'placeholder': '30',
                    'description': 'Frames per second',
                    'required': False,
                    'default': 30
                },
                {
                    'name': 'x',
                    'label': 'X Offset',
                    'type': 'number',
                    'min': 0,
                    'placeholder': '0',
                    'description': 'Horizontal offset in pixels',
                    'required': False
                },
                {
                    'name': 'y',
                    'label': 'Y Offset',
                    'type': 'number',
                    'min': 0,
                    'placeholder': '0',
                    'description': 'Vertical offset in pixels',
                    'required': False
                }
            ]
        }

    def start_async(self):
        """
        Start background thread to continuously capture frames from a Genicam compliant camera.
        """
        import threading
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
        while not self._stop_event.is_set():  # type: ignore
            success, frame = self._read_direct()
            if success:
                self._latest_frame = frame
                # print(self._latest_frame.mean())

            time.sleep(1 / self.fps)

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the most recent frame captured by the background thread.
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        frame = getattr(self, '_latest_frame', None)
        return (frame is not None), frame

    @staticmethod
    def buffer_to_numpy(buffer):
        from harvesters.util.pfnc import mono_location_formats, \
            rgb_formats, bgr_formats, \
            rgba_formats, bgra_formats, bayer_location_formats, lmn_422_location_formats, \
            lmn_422_packed_location_formats, lmn_411_location_formats
        import cv2

        payload = buffer.payload
        component = payload.components[0]
        width = component.width
        height = component.height
        data_format = component.data_format

        if data_format in mono_location_formats:
            content = component.data.reshape(height, width)
        else:
            if data_format in rgb_formats or \
                    data_format in rgba_formats or \
                    data_format in bgr_formats or \
                    data_format in bgra_formats or \
                    data_format in bayer_location_formats:

                content = component.data.reshape(
                    height, width,
                    int(component.num_components_per_pixel)
                )

                if data_format in bayer_location_formats:
                    content = cv2.cvtColor(content, cv2.COLOR_BayerGR2RGB)

                if data_format in rgb_formats:
                    content = content[:, :, ::-1]
            elif data_format in lmn_422_location_formats or \
                    data_format in lmn_422_packed_location_formats or \
                    data_format in lmn_411_location_formats:

                ycbcr422_data = component.data.reshape((-1, 4))

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
            else:
                raise NotImplementedError(f"Unsupported pixel data format `{data_format}`")
        return content

    def _read_direct(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Directly read a frame from the Genicam compliant camera (bypassing background thread logic).
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        from genicam.gentl import TimeoutException

        if not self.is_connected or self.camera is None:
            return False, None
        try:
            n = self.camera.remote_device.node_map
            n.TriggerSoftware.execute()

            self.camera = self.camera
            buffer = self.camera.fetch(timeout=3)
            nump = self.buffer_to_numpy(buffer)

            buffer.queue()

            return True, nump
        except TimeoutException:
            # We have an ImageAcquirer object but nothing has
            # been fetched, wait for the next round:
            return False, None
        except Exception as e:
            logger.error(f"Error reading from Genicam camera: {e}")
            return False, None

    """Genicam camera capture using genicam."""
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
        self.fps = self.config.get("fps",60)

    def try_set_node_param(self, container, param_name, attr_name, value):
        try:
            param = getattr(container, param_name)
            setattr(param, attr_name, value)
            print(f"Set {param_name}.{attr_name} to {value}")
        except Exception as e:
            print(f"Failed to set {param_name} to {value}: {e}")

    def connect(self) -> bool:
        """Connect to Genicam camera."""
        if self.h is None:
            logger.error("Harvesters not available")
            return False

        try:
            # self.h.add_file('/opt/pylon/lib/gentlproducer/gtl/ProducerU3V.cti')

            cti_files = self.config.get("cti_files",[])
            for cti_file in cti_files:
                self.h.add_file(cti_file)
            self.h.update()

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

            # Try to configure the camera in more suitable settings
            self.try_set_node_param(n, "TriggerMode", "value", "On")
            self.try_set_node_param(n, "TriggerActivation", "value", "RisingEdge")
            self.try_set_node_param(n, "TriggerSource", "value", "Software")
            self.try_set_node_param(n, "PixelFormat", "value", "BayerGR8")
            self.try_set_node_param(n, "BinningHorizontal", "value", 1)
            self.try_set_node_param(n, "BinningVertical", "value", 1)

            # from genicam_tools import GenicamTools
            # node_map = GenicamTools.print_node_map(n)

            self.is_connected = True

            # Apply config parameters
            if 'exposure' in self.config:
                self.set_exposure(self.config['exposure'])
            if 'gain' in self.config:
                self.set_gain(self.config['gain'])
            if 'width' in self.config and 'height' in self.config:
                self.set_frame_size(self.config['width'], self.config['height'])
            if 'x' in self.config or 'y' in self.config:
                self.set_offset(self.config.get('x', 0), self.config.get('y', 0))
            if 'fps' in self.config:
                self.fps = self.config['fps']
            if 'acquisition_framerate' in self.config:
                self.set_fps(self.config['acquisition_framerate'])

            self.camera.start()

            logger.info(f"Connected to Genicam camera {self.camera.device.module.vendor} {self.camera.device.module.model}")
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
            n = self.camera.remote_device.node_map

            min_exposure = n.ExposureTime.min
            max_exposure = n.ExposureTime.max
            return (min_exposure, max_exposure)
        except Exception as e:
            logger.error(f"Error getting exposure range: {e}")
            return (0.0, 0.0)

    def get_gain_range(self) -> Tuple[float, float]:
        """Get gain range in dB."""
        if not self.is_connected or self.camera is None:
            return (0.0, 0.0)

        try:
            n = self.camera.remote_device.node_map

            min_gain = n.Gain.min
            max_gain = n.Gain.max
            return (min_gain, max_gain)
        except Exception as e:
            logger.error(f"Error getting gain range: {e}")
            return (0.0, 0.0)

    def set_exposure(self, value: float) -> bool:
        """Set exposure in microseconds."""
        if not self.is_connected or self.camera is None:
            return False

        try:
            n = self.camera.remote_device.node_map

            n.ExposureTime.value = value
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
            n = self.camera.remote_device.node_map

            return n.ExposureTime.value
        except Exception:
            return self._exposure

    def set_gain(self, value: float) -> bool:
        """Set gain in dB."""
        if not self.is_connected or self.camera is None:
            return False
        try:
            n = self.camera.remote_device.node_map
            n.Gain.value = value

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
            n = self.camera.remote_device.node_map

            return n.Gain.value
        except Exception:
            return self._gain

    def enable_auto_exposure(self, enable: bool = True) -> bool:
        """Enable or disable auto exposure for Genicam camera."""
        if not self.is_connected or self.camera is None:
            return False
        try:
            n = self.camera.remote_device.node_map
            if enable:
                n.ExposureAuto.value = "Continuous"
                n.GainAuto.value = "Continuous"
            else:
                n.ExposureAuto.value = "Off"
                n.GainAuto.value = "Off"
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
            n = self.camera.remote_device.node_map

            n.Width.value = width
            n.Height.value = height
            logger.info(f"Set Genicam camera resolution to {width}x{height}")
            return True
        except Exception as e:
            logger.error(f"Error setting Genicam camera resolution: {e}")
            return False

    def set_offset(self, x: int, y: int) -> bool:
        """Set offset for Genicam camera."""
        if not self.is_connected or self.camera is None:
            return False
        try:
            n = self.camera.remote_device.node_map

            n.OffsetX.value = x
            n.OffsetY.value = y
            logger.info(f"Set Genicam offset to ({x},{y})")
            return True
        except Exception as e:
            logger.error(f"Error setting Genicam camera offset: {e}")
            return False

    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        """Get frame size."""
        if not self.is_connected or self.camera is None:
            return None

        try:
            n = self.camera.remote_device.node_map

            width = n.Width.value
            height = n.Height.value
            return (width, height)
        except Exception:
            return None

    def set_fps(self, fps: float) -> bool:
        """Set FPS for Genicam camera."""
        if not self.is_connected or self.camera is None:
            return False
        try:
            n = self.camera.remote_device.node_map

            n.AcquisitionFrameRateEnable.value = True
            n.AcquisitionFrameRate.value = fps
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
            n = self.camera.remote_device.node_map
            return n.AcquisitionFrameRate.value
        except Exception:
            return None

    @classmethod
    def discover(cls) -> list:
        """
        Discover available GenICam compliant cameras.
        
        Returns:
            list: List of dictionaries containing GenICam camera information.
                Each dict contains: {'index': int, 'serial_number': str, 'model_name': str, 'vendor': str}
        """
        devices = []
        
        try:
            from harvesters.core import Harvester
        except ImportError:
            logger.warning("Harvesters module not available. Cannot discover GenICam cameras.")
            return []
        
        harvester = None
        try:
            harvester = Harvester()
            
            # Add common GenTL producer paths (this may need customization)
            try:
                # Try to add some common GenTL producers
                harvester.add_cti_file('/opt/pylon5/lib64/pylon_TL_GenICam.cti')  # Basler
                harvester.add_cti_file('/opt/mvIMPACT_acquire/lib/x86_64/mvGenTLProducer.cti')  # MATRIX VISION
            except:
                pass  # If paths don't exist, that's fine
            
            harvester.update()
            
            for i, device_info in enumerate(harvester.device_info_list):
                try:
                    device_data = {
                        'index': i,
                        'serial_number': getattr(device_info, 'serial_number', f'genicam_{i}'),
                        'model_name': getattr(device_info, 'model', 'GenICam Camera'),
                        'vendor': getattr(device_info, 'vendor', 'Unknown')
                    }
                    devices.append(device_data)
                    logger.info(f"Found GenICam camera: {device_data}")
                    
                except Exception as e:
                    logger.warning(f"Could not get info for GenICam device {i}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error discovering GenICam cameras: {e}")
        finally:
            if harvester:
                try:
                    harvester.reset()
                except:
                    pass
        
        return devices


if __name__ == "__main__":
    # Example usage
    import cv2
    
    devices = GenicamCapture.discover()
    print("Discovered GenICam cameras:")
    for device in devices:
        print(f"  - {device['model_name']} (Serial: {device['serial_number']})")

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
