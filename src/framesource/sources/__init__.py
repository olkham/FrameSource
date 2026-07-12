"""Frame source capture backends for FrameSource.

This package contains the individual capture implementations. Most users should
use :class:`framesource.FrameSourceFactory` rather than importing these directly.
Optional backends (vendor SDKs) are imported lazily and may be unavailable.
"""

import importlib
import logging

from .video_capture_base import VideoCaptureBase

logger = logging.getLogger(__name__)

__all__ = ["VideoCaptureBase"]

# Map of public class name -> module within this package. Optional/vendor
# backends are imported defensively so missing SDKs don't break the package.
_OPTIONAL_CLASSES = {
    "WebcamCapture": "webcam_capture",
    "IPCameraCapture": "ipcamera_capture",
    "BaslerCapture": "basler_capture",
    "GenicamCapture": "genicam_capture",
    "RealsenseCapture": "realsense_capture",
    "XimeaCapture": "ximea_capture",
    "HuatengCapture": "huateng_capture",
    "VideoFileCapture": "video_file_capture",
    "FolderCapture": "folder_capture",
    "ScreenCapture": "screen_capture",
    "AudioSpectrogramCapture": "audiospectrogram_capture",
}

for _class_name, _module_name in _OPTIONAL_CLASSES.items():
    try:
        _module = importlib.import_module(f".{_module_name}", __name__)
        globals()[_class_name] = getattr(_module, _class_name)
        __all__.append(_class_name)
    except Exception as e:  # noqa: BLE001 - vendor SDKs may raise non-ImportError
        globals()[_class_name] = None
        logger.debug("%s unavailable: %s", _class_name, e)
