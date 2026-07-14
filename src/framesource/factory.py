"""
Video Capture System with Factory Pattern

A comprehensive video capture system that supports multiple backends:
- Webcam (OpenCV)
- IP Camera (RTSP/HTTP)
- Industrial cameras (Basler, GenICam)
- Custom capture APIs

Usage:
    capture = FrameSourceFactory.create('webcam', source=0)
    capture.connect()
    capture.set_exposure(50)
    frame = capture.read()
"""

import importlib
import json
import logging
import os
import warnings
from pathlib import Path
from typing import Any, Literal, Optional, Union

from .errors import MissingDependencyError, UnknownSourceTypeError
from .sources.video_capture_base import VideoCaptureBase

logger = logging.getLogger(__name__)


def _warn_deprecated_param(old: str, new: str) -> None:
    """Log and warn that a factory parameter name is deprecated."""
    message = f"Parameter '{old}' is deprecated, use '{new}' instead"
    logger.warning(message)
    warnings.warn(message, DeprecationWarning, stacklevel=3)


# Import capture classes with graceful handling for missing optional
# dependencies (e.g. vendor SDKs). Each entry maps a factory key to its module
# and class name; sources that fail to import are simply omitted.
_OPTIONAL_SOURCES = [
    ("webcam", "webcam_capture", "WebcamCapture"),
    ("ipcam", "ipcamera_capture", "IPCameraCapture"),
    ("basler", "basler_capture", "BaslerCapture"),
    ("genicam", "genicam_capture", "GenicamCapture"),
    ("realsense", "realsense_capture", "RealsenseCapture"),
    ("ximea", "ximea_capture", "XimeaCapture"),
    ("huateng", "huateng_capture", "HuatengCapture"),
    ("video_file", "video_file_capture", "VideoFileCapture"),
    ("folder", "folder_capture", "FolderCapture"),
    ("screen", "screen_capture", "ScreenCapture"),
    ("audio_spectrogram", "audiospectrogram_capture", "AudioSpectrogramCapture"),
]

_capture_imports: dict[str, type] = {}

for _key, _module_name, _class_name in _OPTIONAL_SOURCES:
    try:
        _module = importlib.import_module(f".sources.{_module_name}", __package__)
        _capture_imports[_key] = getattr(_module, _class_name)
    except Exception as e:  # noqa: BLE001 - vendor SDKs may raise non-ImportError
        logger.debug("%s unavailable: %s", _class_name, e)


class FrameSourceFactory:
    """Factory class for creating video capture instances."""

    MediaSource = Literal[
        "folder",
        "video_file",
        "webcam",
        "ipcam",
        "basler",
        "realsense",
        "screen",
        "genicam",
        "ximea",
        "huateng",
        "audio_spectrogram",
    ]

    CameraSource = Literal["webcam", "realsense", "genicam", "basler", "ximea", "huateng"]

    _capture_types: dict[str, type] = _capture_imports

    @classmethod
    def create(
        cls,
        source_type: Any = None,
        source_id: Any = None,
        capture_type: Any = None,
        source: Any = None,
        **kwargs,
    ) -> VideoCaptureBase:
        """
        Create a video capture instance.

        Args:
            source_type: Type of capture ('webcam', 'ipcam', 'basler', 'genicam', 'custom')
            source_id: Source identifier
            capture_type: (DEPRECATED) Use source_type instead
            source: (DEPRECATED) Use source_id instead
            connect: If True (default), connect() is called before returning.
            **kwargs: Additional parameters for the specific capture type

        Returns:
            VideoCaptureBase: Configured capture instance

        Raises:
            UnknownSourceTypeError: If source_type is not supported (also
                catchable as ValueError for backwards compatibility).
        """
        # Handle backward compatibility for capture_type -> source_type
        if source_type is None and capture_type is not None:
            _warn_deprecated_param("capture_type", "source_type")
            source_type = capture_type

        # Try to get source_type from kwargs if still not set
        if source_type is None:
            source_type = kwargs.pop("source_type", None)

        # Fall back to capture_type in kwargs for backward compatibility
        if source_type is None:
            source_type = kwargs.pop("capture_type", None)
            if source_type is not None:
                _warn_deprecated_param("capture_type", "source_type")

        if not source_type or source_type not in cls._capture_types:
            available_types = ", ".join(cls._capture_types.keys())
            raise UnknownSourceTypeError(
                f"Unsupported capture type: {source_type}. Available types: {available_types}"
            )

        # Handle backward compatibility for source -> source_id
        if source_id is None and source is not None:
            _warn_deprecated_param("source", "source_id")
            source_id = source

        # Try to get source_id from kwargs if still not set
        if source_id is None:
            source_id = kwargs.pop("source_id", None)

        # Fall back to source in kwargs for backward compatibility
        if source_id is None:
            source_id = kwargs.pop("source", None)
            if source_id is not None:
                _warn_deprecated_param("source", "source_id")

        if source_id is None:
            logger.warning("Source not provided, defaulting to 0")
            source_id = 0

        # Pop control kwargs before they reach the capture constructor.
        connect = kwargs.pop("connect", True)

        capture_class = cls._capture_types[source_type]
        cc = capture_class(source=source_id, **kwargs)

        if connect:
            cc.connect()

        return cc

    @classmethod
    def from_config(cls, config: Union[dict[str, Any], str, os.PathLike]) -> VideoCaptureBase:
        """Create a capture instance from a configuration dict or file.

        The configuration supplies ``source_type`` and (optionally)
        ``source_id`` plus any additional keyword arguments understood by
        :meth:`create` (including ``connect``). Every key other than
        ``source_type`` and ``source_id`` is forwarded to :meth:`create`
        unchanged.

        Args:
            config: Either a mapping of configuration values, or a path (``str``
                or :class:`os.PathLike`) to a ``.json``, ``.yaml`` or ``.yml``
                file containing that mapping. A dict argument is never mutated.

        Returns:
            VideoCaptureBase: The configured capture instance.

        Raises:
            MissingDependencyError: If a ``.yaml``/``.yml`` file is given but
                PyYAML is not installed.
            ValueError: If a file with an unsupported extension is given.
            FileNotFoundError: If the given path does not exist.
            UnknownSourceTypeError: If ``source_type`` is missing or unknown.

        Example:
            ```python
            from framesource import FrameSourceFactory

            # From an in-memory dict:
            cap = FrameSourceFactory.from_config({
                "source_type": "webcam",
                "source_id": 0,
                "connect": False,
            })

            # From a JSON or YAML file:
            cap = FrameSourceFactory.from_config("camera.json")
            ```
        """
        if isinstance(config, (str, os.PathLike)):
            config = cls._read_config_file(config)

        # Shallow-copy so the caller's dict is never mutated.
        params = dict(config)
        source_type = params.pop("source_type", None)
        source_id = params.pop("source_id", None)
        return cls.create(source_type, source_id=source_id, **params)

    @staticmethod
    def _read_config_file(path: Union[str, os.PathLike]) -> dict[str, Any]:
        """Load a config mapping from a ``.json``, ``.yaml`` or ``.yml`` file."""
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix == ".json":
            with open(path, encoding="utf-8") as fh:
                return json.load(fh)
        if suffix in (".yaml", ".yml"):
            try:
                import yaml
            except ImportError as e:
                raise MissingDependencyError(
                    "pyyaml",
                    extra=None,
                    details="required to load .yaml config files",
                ) from e
            with open(path, encoding="utf-8") as fh:
                return yaml.safe_load(fh)
        raise ValueError(
            f"Unsupported config file extension: {path.suffix!r} (expected .json, .yaml or .yml)"
        )

    @classmethod
    def register_capture_type(cls, name: str, capture_class: type):
        """
        Register a new capture type.

        Args:
            name: Name of the capture type
            capture_class: Class implementing VideoCaptureBase
        """
        if not issubclass(capture_class, VideoCaptureBase):
            raise ValueError("Capture class must inherit from VideoCaptureBase")

        if name in cls._capture_types:
            logger.warning(f"Capture type '{name}' already registered, replacing with new class.")

        cls._capture_types[name] = capture_class
        logger.info(f"Registered new capture type: {name}")

    @classmethod
    def discover_devices(cls, sources: Optional[list[str]] = None) -> dict:
        """
        Discover available capture devices from the registered capture types.

        This method queries each capture type for connected devices and
        returns a dictionary mapping source names to their discovery results.

        Args:
            sources (list[str], optional): Specific source keys to limit discovery to.
                If None, all registered capture types are queried.

        Returns:
            dict: A mapping of source keys to the discovered devices for each.
                  Sources that return no devices are excluded.
        """

        _sources = (
            cls._capture_types.items()
            if sources is None
            else ((k, v) for k, v in cls._capture_types.items() if k in sources)
        )

        return {k: ret for k, v in _sources if (ret := v.discover())}

    @classmethod
    def get_available_types(cls) -> list:
        """Get list of available capture types."""
        return list(cls._capture_types.keys())

    @classmethod
    def unregister_capture_type(cls, capture_type: str):
        """Unregister a capture type (convenience function)"""
        if capture_type not in cls._capture_types:
            raise UnknownSourceTypeError(f"Capture type '{capture_type}' is not registered")
        del cls._capture_types[capture_type]
        logger.info(f"Unregistered capture type: {capture_type}")
