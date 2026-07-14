import logging
import platform
import warnings
from typing import Any, Optional, Union

import cv2
import numpy as np

from ..discovery import DeviceInfo
from .video_capture_base import VideoCaptureBase

logger = logging.getLogger(__name__)

# String aliases for OpenCV capture backends, resolved to the cv2.CAP_* ints.
# Only backends present in this OpenCV build are included.
_BACKEND_ALIASES: dict[str, int] = {}
for _alias, _const_name in (
    ("any", "CAP_ANY"),
    ("auto", "CAP_ANY"),
    ("dshow", "CAP_DSHOW"),
    ("msmf", "CAP_MSMF"),
    ("v4l2", "CAP_V4L2"),
    ("avfoundation", "CAP_AVFOUNDATION"),
    ("gstreamer", "CAP_GSTREAMER"),
    ("ffmpeg", "CAP_FFMPEG"),
):
    if hasattr(cv2, _const_name):
        _BACKEND_ALIASES[_alias] = getattr(cv2, _const_name)

# Uncompressed pixel formats. When one of these is negotiated at high
# resolution the USB link is easily saturated, throttling the frame rate far
# below what MJPG (compressed) would sustain.
_UNCOMPRESSED_FOURCCS = frozenset(
    {"YUY2", "YUYV", "UYVY", "I420", "IYUV", "YV12", "NV12", "RGB3", "BGR3"}
)

# Resolutions at or above this pixel count are where uncompressed transport
# starts to hurt (720p and up).
_HIGH_RES_PIXELS = 1280 * 720


class WebcamCapture(VideoCaptureBase):
    has_discovery = True
    supports_exposure = True
    supports_gain = True

    """Webcam capture using OpenCV."""

    def __init__(
        self,
        source: int = 0,
        *,
        width: Optional[int] = None,
        height: Optional[int] = None,
        fps: Optional[float] = None,
        backend: Optional[Union[str, int]] = None,
        fourcc: Optional[str] = None,
        **kwargs,
    ):
        """Initialize the webcam capture.

        Args:
            source: Camera device index, or an ``api_pref:index`` /
                ``api_pref:path`` string (as produced in the ``id`` field of
                :meth:`discover`).
            width: Desired frame width in pixels. Applied on :meth:`connect`
                (together with ``height``) when provided.
            height: Desired frame height in pixels. Applied on
                :meth:`connect` (together with ``width``) when provided.
            fps: Desired frames per second. Applied on :meth:`connect` when
                provided.
            backend: OpenCV capture backend to open the device with. Accepts a
                name (``'msmf'``, ``'dshow'``, ``'v4l2'``, ``'avfoundation'``,
                ``'gstreamer'``, ``'ffmpeg'``, ``'any'``) or a raw
                ``cv2.CAP_*`` integer. ``None`` (default) uses the OS default
                (DirectShow on Windows, AVFoundation on macOS, V4L2 on Linux).
                Note that on Windows ``'dshow'`` and ``'msmf'`` interpret the
                exposure/gain values in :meth:`set_exposure`/:meth:`set_gain`
                differently. If a high-resolution stream is stuck at a low
                frame rate, ``'msmf'`` will often negotiate a compressed format
                where DirectShow will not.
            fourcc: Desired pixel format as a four-character code, e.g.
                ``'MJPG'``. Applied on :meth:`connect` *before* the resolution
                so it is not reset by the resolution change. ``None`` (default)
                leaves the backend's default format untouched. Requesting
                ``'MJPG'`` is the usual fix for uncompressed formats saturating
                the USB link at 1080p and above (see the warning logged by
                :meth:`connect`). Not all backends honour every code.
            **kwargs: Additional passthrough options stored on ``self.config``.
        """
        # Preserve the historical ``self.config`` contents: these options were
        # previously mined from ``**kwargs`` and read back via ``self.config``
        # in ``connect()``. Only forward explicitly-provided values so the
        # ``'width' in self.config`` presence checks keep their old meaning.
        for _key, _val in (("width", width), ("height", height), ("fps", fps)):
            if _val is not None:
                kwargs[_key] = _val
        super().__init__(source, **kwargs)
        self.cap = None
        self.fourcc = fourcc
        self.api_preference = self._resolve_backend(backend)

        self.source = source

        if "is_mono" in kwargs:
            logger.warning(
                "'is_mono' argument is only used for certain industrial cameras "
                "and has no effect for webcams."
            )

    @staticmethod
    def _resolve_backend(backend: Optional[Union[str, int]]) -> int:
        """Resolve a backend name/int to a cv2.CAP_* constant.

        ``None`` maps to the OS default; an int is used verbatim; a string is
        looked up (case-insensitively) in :data:`_BACKEND_ALIASES`.
        """
        if backend is None:
            if platform.system() == "Windows":
                return cv2.CAP_DSHOW
            elif platform.system() == "Darwin":
                return cv2.CAP_AVFOUNDATION
            else:
                return cv2.CAP_V4L2
        if isinstance(backend, int):
            return backend
        key = backend.strip().lower()
        if key not in _BACKEND_ALIASES:
            available = ", ".join(sorted(_BACKEND_ALIASES))
            raise ValueError(
                f"Unknown webcam backend {backend!r}. Use one of: {available}, "
                f"or pass a cv2.CAP_* integer."
            )
        return _BACKEND_ALIASES[key]

    def connect(self) -> bool:
        """Connect to webcam."""
        try:
            src = self.source
            api_pref = self.api_preference
            # Support `api_pref/index` format or `api_pref/path` format used in
            # list_devices `id` field
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

            # Pixel format must be set before the resolution: several drivers
            # reset the format back to their default when the frame size
            # changes.
            if self.fourcc:
                self._apply_fourcc(self.fourcc)

            # Set additional parameters if provided
            if "width" in self.config and "height" in self.config:
                self.set_frame_size(self.config["width"], self.config["height"])
            if "fps" in self.config:
                self.cap.set(cv2.CAP_PROP_FPS, self.config["fps"])

            self.is_connected = True
            self._warn_if_bandwidth_limited()
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

    def _read_implementation(self) -> tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the webcam.
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self.is_connected or self.cap is None:
            return False, None
        ret, frame = self.cap.read()
        return ret, frame if ret else None

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

    def get_frame_size(self) -> Optional[tuple[int, int]]:
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

    def _apply_fourcc(self, fourcc: str) -> bool:
        """Request a pixel format from the device by its four-character code.

        Returns True if OpenCV accepted the request. Note that acceptance does
        not guarantee the driver actually switches format (see
        :meth:`get_fourcc` / :meth:`_warn_if_bandwidth_limited`).
        """
        if self.cap is None:
            return False
        if len(fourcc) != 4:
            logger.warning("Ignoring invalid fourcc %r (must be 4 characters, e.g. 'MJPG')", fourcc)
            return False
        # VideoWriter_fourcc moved to VideoWriter.fourcc in newer OpenCV builds.
        fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None) or cv2.VideoWriter.fourcc
        code = fourcc_fn(*fourcc)
        ok = bool(self.cap.set(cv2.CAP_PROP_FOURCC, code))
        logger.info("Requested pixel format %s (accepted: %s)", fourcc, ok)
        return ok

    def get_fourcc(self) -> Optional[str]:
        """Return the currently negotiated pixel format as a 4-char code.

        Returns None if not connected. Some backends (notably MSMF) report an
        empty string even while delivering a compressed stream.
        """
        if not self.is_connected or self.cap is None:
            return None
        value = int(self.cap.get(cv2.CAP_PROP_FOURCC))
        return "".join(chr((value >> (8 * i)) & 0xFF) for i in range(4)).strip("\x00 ")

    def _warn_if_bandwidth_limited(self) -> None:
        """Log a warning when a high-res stream negotiated an uncompressed format.

        This is the common cause of a webcam that requests 1080p30 but only
        delivers a handful of frames per second: an uncompressed format (e.g.
        YUY2) saturates the USB link. Requesting ``fourcc='MJPG'`` or a
        different ``backend`` usually resolves it.
        """
        size = self.get_frame_size()
        if not size:
            return
        width, height = size
        if width * height < _HIGH_RES_PIXELS:
            return
        negotiated = self.get_fourcc()
        # MSMF reports an empty code even on a fast compressed stream, so only
        # warn when we positively identify an uncompressed format.
        if negotiated and negotiated.upper() in _UNCOMPRESSED_FOURCCS:
            logger.warning(
                "Webcam negotiated uncompressed %s at %dx%d; USB bandwidth may cap the "
                "frame rate well below the requested value. Pass fourcc='MJPG', or try a "
                "different backend (e.g. backend='msmf' on Windows).",
                negotiated,
                width,
                height,
            )

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
    def discover(cls) -> list[DeviceInfo]:
        try:
            devices: list[DeviceInfo] = []
            from cv2.videoio_registry import getBackendName
            from cv2_enumerate_cameras import enumerate_cameras

            camera_list = []
            if platform.system() == "Windows":
                backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF]
            elif platform.system() == "Darwin":  # macOS
                backends = [cv2.CAP_AVFOUNDATION]
            else:  # Linux
                backends = [cv2.CAP_V4L2]

            for backend in backends:
                camera_list.extend(enumerate_cameras(backend))

            # Sort camera list by index
            camera_list.sort(key=lambda cam: cam.index)

            for camera_info in camera_list:
                # OS enumeration index -> not a stable identifier across replug.
                devices.append(
                    DeviceInfo(
                        device_id=f"{camera_info.backend}:{camera_info.index}:{camera_info.path}",
                        index=camera_info.index,
                        name=camera_info.name,
                        driver="opencv",
                        id_stable=False,
                        metadata={
                            "backend_index": camera_info.backend,
                            "backend_name": getBackendName(camera_info.backend),
                        },
                    )
                )
            logger.info(f"Found {cls.__name__} input device: {devices}")
            return devices
        except ImportError:
            logger.warning(
                "cv2-enumerate-cameras module not available. Install cv2-enumerate-cameras "
                "to list available (web)cameras."
            )
        return []

    @classmethod
    def get_config_schema(cls) -> dict[str, Any]:
        """Get configuration schema for webcam capture"""
        warnings.warn(
            "get_config_schema() is deprecated and will be removed in a future release; "
            "UI form schemas belong in the consuming application.",
            DeprecationWarning,
            stacklevel=2,
        )
        return {
            "title": "Webcam Configuration",
            "description": "Configure USB webcam or built-in camera settings",
            "fields": [
                {
                    "name": "source",
                    "label": "Camera Index",
                    "type": "number",
                    "min": 0,
                    "max": 10,
                    "placeholder": "0",
                    "description": "Camera device index (0 for default, 1 for second camera, etc.)",
                    "required": False,
                    "default": 0,
                },
                {
                    "name": "width",
                    "label": "Width",
                    "type": "number",
                    "min": 160,
                    "max": 4096,
                    "placeholder": "1920",
                    "description": "Frame width in pixels",
                    "required": False,
                },
                {
                    "name": "height",
                    "label": "Height",
                    "type": "number",
                    "min": 120,
                    "max": 2160,
                    "placeholder": "1080",
                    "description": "Frame height in pixels",
                    "required": False,
                },
                {
                    "name": "fps",
                    "label": "Frame Rate (FPS)",
                    "type": "number",
                    "min": 1,
                    "max": 120,
                    "placeholder": "30",
                    "description": "Frames per second",
                    "required": False,
                    "default": 30,
                },
                {
                    "name": "exposure",
                    "label": "Exposure",
                    "type": "number",
                    "min": -13,
                    "max": -1,
                    "step": 1,
                    "placeholder": "-6",
                    "description": "Manual exposure value (-13 to -1, lower = brighter)",
                    "required": False,
                },
                {
                    "name": "gain",
                    "label": "Gain",
                    "type": "number",
                    "min": 0,
                    "max": 255,
                    "placeholder": "0",
                    "description": "Camera gain (0-255)",
                    "required": False,
                },
            ],
        }


if __name__ == "__main__":
    # Example usage

    print("Discovering webcams...")
    webcams = WebcamCapture.discover()
    print(f"\nFound {len(webcams)} webcam(s):\n")
    for cam in webcams:
        print(f"  [{cam['index']}] {cam['name']}")

    if webcams:
        print("\nTesting first camera...")
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
                    cv2.imshow("Webcam", frame)  # type: ignore
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                else:
                    print("Failed to read frame from webcam.")
                    break

            camera.disconnect()
        else:
            print("Failed to connect to webcam.")
    else:
        print("\nNo webcams found.")
