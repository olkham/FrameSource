import logging
import os
import warnings
from typing import Any, Optional

import numpy as np

from ..errors import MissingDependencyError
from .video_capture_base import VideoCaptureBase

logger = logging.getLogger(__name__)


class GenicamCapture(VideoCaptureBase):
    has_discovery = True
    supports_exposure = True
    supports_gain = True

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Get configuration schema for GenICam capture"""
        warnings.warn(
            "get_config_schema() is deprecated and will be removed in a future release; "
            "UI form schemas belong in the consuming application.",
            DeprecationWarning,
            stacklevel=2,
        )
        return {
            "title": "GenICam Camera Configuration",
            "description": "Configure GenICam compliant camera settings",
            "fields": [
                {
                    "name": "source",
                    "label": "Camera Source",
                    "type": "text",
                    "placeholder": "Serial number or device index (0, 1, 2...)",
                    "description": "Camera serial number or device index",
                    "required": False,
                    "default": "0",
                },
                {
                    "name": "cti_files",
                    "label": "CTI Files",
                    "type": "text",
                    "placeholder": "/path/to/producer.cti",
                    "description": "GenTL producer files (comma-separated)",
                    "required": False,
                },
                {
                    "name": "exposure",
                    "label": "Exposure Time (µs)",
                    "type": "number",
                    "min": 1,
                    "max": 1000000,
                    "placeholder": "10000",
                    "description": "Exposure time in microseconds",
                    "required": False,
                },
                {
                    "name": "gain",
                    "label": "Gain (dB)",
                    "type": "number",
                    "min": 0,
                    "max": 40,
                    "step": 0.1,
                    "placeholder": "0.0",
                    "description": "Camera gain in decibels",
                    "required": False,
                },
                {
                    "name": "width",
                    "label": "Width",
                    "type": "number",
                    "min": 1,
                    "max": 10000,
                    "placeholder": "1920",
                    "description": "Frame width in pixels",
                    "required": False,
                },
                {
                    "name": "height",
                    "label": "Height",
                    "type": "number",
                    "min": 1,
                    "max": 10000,
                    "placeholder": "1080",
                    "description": "Frame height in pixels",
                    "required": False,
                },
                {
                    "name": "fps",
                    "label": "Frame Rate (FPS)",
                    "type": "number",
                    "min": 1,
                    "max": 240,
                    "placeholder": "30",
                    "description": "Frames per second",
                    "required": False,
                    "default": 30,
                },
                {
                    "name": "x",
                    "label": "X Offset",
                    "type": "number",
                    "min": 0,
                    "placeholder": "0",
                    "description": "Horizontal offset in pixels",
                    "required": False,
                },
                {
                    "name": "y",
                    "label": "Y Offset",
                    "type": "number",
                    "min": 0,
                    "placeholder": "0",
                    "description": "Vertical offset in pixels",
                    "required": False,
                },
            ],
        }

    @staticmethod
    def buffer_to_numpy(buffer):
        import cv2
        from harvesters.util.pfnc import (
            bayer_location_formats,
            bgr_formats,
            bgra_formats,
            lmn_411_location_formats,
            lmn_422_location_formats,
            lmn_422_packed_location_formats,
            mono_location_formats,
            rgb_formats,
            rgba_formats,
        )

        payload = buffer.payload
        component = payload.components[0]
        width = component.width
        height = component.height
        data_format = component.data_format

        if data_format in mono_location_formats:
            content = component.data.reshape(height, width)
        else:
            if (
                data_format in rgb_formats
                or data_format in rgba_formats
                or data_format in bgr_formats
                or data_format in bgra_formats
                or data_format in bayer_location_formats
            ):
                content = component.data.reshape(
                    height, width, int(component.num_components_per_pixel)
                )

                if data_format in bayer_location_formats:
                    content = cv2.cvtColor(content, cv2.COLOR_BayerGR2RGB)

                if data_format in rgb_formats:
                    content = content[:, :, ::-1]
            elif (
                data_format in lmn_422_location_formats
                or data_format in lmn_422_packed_location_formats
                or data_format in lmn_411_location_formats
            ):
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

    def _read_implementation(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the Genicam compliant camera.
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

    def __init__(
        self,
        source: Any = None,
        *,
        is_mono: Optional[bool] = None,
        cti_files: Optional[list[str]] = None,
        exposure: Optional[float] = None,
        gain: Optional[float] = None,
        width: Optional[int] = None,
        height: Optional[int] = None,
        x: Optional[int] = None,
        y: Optional[int] = None,
        fps: Optional[float] = None,
        acquisition_framerate: Optional[float] = None,
        **kwargs,
    ):
        """Initialize the GenICam capture.

        Args:
            source: Camera serial number (str) or device index (int).
            is_mono: Stored for parity with other vendor sources (default:
                False). GenICam pixel format is currently negotiated
                automatically in :meth:`connect`, so this flag has no
                effect yet.
            cti_files: GenTL producer (``.cti``) file paths to load, in
                addition to any discovered via the ``GENICAM_GENTL64_PATH``
                environment variable. Applied on :meth:`connect` when
                provided.
            exposure: Initial exposure time in microseconds. Applied on
                :meth:`connect` when provided.
            gain: Initial gain in dB. Applied on :meth:`connect` when
                provided.
            width: Desired frame width in pixels. Applied on :meth:`connect`
                (together with ``height``) when provided.
            height: Desired frame height in pixels. Applied on
                :meth:`connect` (together with ``width``) when provided.
            x: Desired horizontal offset in pixels. Applied on
                :meth:`connect` when ``x`` or ``y`` is provided.
            y: Desired vertical offset in pixels. Applied on :meth:`connect`
                when ``x`` or ``y`` is provided.
            fps: Target frame rate, stored on ``self.fps`` (default: 60 when
                not provided).
            acquisition_framerate: Frame rate applied via :meth:`set_fps` on
                :meth:`connect` when provided.
            **kwargs: Additional passthrough options stored on ``self.config``.

        Raises:
            MissingDependencyError: If the ``harvesters`` package is not
                installed.
        """
        # Preserve the historical ``self.config`` contents: only forward
        # explicitly-provided values so the ``'width' in self.config`` style
        # presence checks in connect() keep their old meaning.
        for _key, _val in (
            ("is_mono", is_mono),
            ("cti_files", cti_files),
            ("exposure", exposure),
            ("gain", gain),
            ("width", width),
            ("height", height),
            ("x", x),
            ("y", y),
            ("fps", fps),
            ("acquisition_framerate", acquisition_framerate),
        ):
            if _val is not None:
                kwargs[_key] = _val
        super().__init__(source, **kwargs)
        self.camera = None
        self.converter = None
        try:
            from harvesters.core import Harvester

            self.h = Harvester()
        except ImportError as e:
            raise MissingDependencyError("harvesters", extra="genicam", details=str(e)) from e

        self.is_mono = self.config.get("is_mono", False)
        self.serial_number = source if isinstance(source, str) else None
        self.device_index = source if isinstance(source, int) else 0
        self.fps = self.config.get("fps", 60)

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

            cti_files = self.config.get("cti_files", [])
            for cti_file in cti_files:
                self.h.add_file(cti_file)
            for file in GenicamCapture.find_cti_paths():
                self.h.add_file(file)

            self.h.update()

            devices = self.h.device_info_list

            if len(devices) == 0:
                logger.error("No Genicam cameras found")
                return False

            # Create camera object
            if self.serial_number:
                self.camera = self.h.create({"serial_number": self.serial_number})
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
            if "exposure" in self.config:
                self.set_exposure(self.config["exposure"])
            if "gain" in self.config:
                self.set_gain(self.config["gain"])
            if "width" in self.config and "height" in self.config:
                self.set_frame_size(self.config["width"], self.config["height"])
            if "x" in self.config or "y" in self.config:
                self.set_offset(self.config.get("x", 0), self.config.get("y", 0))
            if "fps" in self.config:
                self.fps = self.config["fps"]
            if "acquisition_framerate" in self.config:
                self.set_fps(self.config["acquisition_framerate"])

            self.camera.start()

            logger.info(
                f"Connected to Genicam camera {self.camera.device.module.vendor} "
                f"{self.camera.device.module.model}"
            )
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

    def get_exposure_range(self) -> tuple[float, float]:
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

    def get_gain_range(self) -> tuple[float, float]:
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

    def get_frame_size(self) -> Optional[tuple[int, int]]:
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

    @staticmethod
    def find_cti_paths():
        res = []
        cti_paths = os.getenv("GENICAM_GENTL64_PATH")
        if cti_paths and len(cti_paths) > 0:
            files = os.listdir(cti_paths)
            for f in files:
                if f.endswith(".cti"):
                    res.append(os.path.join(cti_paths, f))
        return res

    @classmethod
    def discover(cls) -> list:
        """
        Discover available GenICam compliant cameras.

        Returns:
            list: List of dictionaries containing GenICam camera information.
                Each dict contains: {'index': int, 'serial_number': str, 'name': str, 'vendor': str}
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
                for file in GenicamCapture.find_cti_paths():
                    harvester.add_file(file)

                # Try to add some common GenTL producers
                harvester.add_file("/opt/pylon5/lib64/pylon_TL_GenICam.cti")  # Basler
                harvester.add_file(
                    "/opt/mvIMPACT_acquire/lib/x86_64/mvGenTLProducer.cti"
                )  # MATRIX VISION
            except Exception:
                pass  # If paths don't exist, that's fine

            harvester.update()

            for i, device_info in enumerate(harvester.device_info_list):
                try:
                    serial = getattr(device_info, "serial_number", f"genicam_{i}")
                    device_data = {
                        "index": i,
                        "id": i,
                        "serial_number": serial,
                        "name": "Genicam " + getattr(device_info, "model", "GenICam Camera"),
                        "vendor": getattr(device_info, "vendor", "Unknown"),
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
                except Exception:
                    pass

        return devices


if __name__ == "__main__":
    # Example usage
    import cv2

    devices = GenicamCapture.discover()
    print("Discovered GenICam cameras:")
    for device in devices:
        print(f"  - {device['name']} (Serial: {device['serial_number']})")

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
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
        camera.disconnect()
    else:
        print("Failed to connect to webcam.")
