"""Deprecated compatibility shim for the old ``frame_source`` package.

This package was renamed to :mod:`framesource`. Importing ``frame_source``
continues to work for now but emits a :class:`DeprecationWarning` and will be
removed in a future major release. Update imports to ``framesource``.
"""

import importlib
import sys
import warnings

warnings.warn(
    "'frame_source' is deprecated and will be removed in a future major "
    "release; import from 'framesource' instead.",
    DeprecationWarning,
    stacklevel=2,
)

import framesource as _framesource
from framesource import *  # noqa: F401,F403
from framesource import FrameSourceFactory, VideoCaptureBase  # noqa: F401

__all__ = getattr(_framesource, "__all__", [])
__version__ = getattr(_framesource, "__version__", "0.0.0")

# Alias the old flat module paths (e.g. ``frame_source.webcam_capture``) onto
# the new locations so existing deep imports keep working.
_SUBMODULE_ALIASES = {
    "factory": "framesource.factory",
    "threading_utils": "framesource.threading_utils",
    "video_capture_base": "framesource.sources.video_capture_base",
    "webcam_capture": "framesource.sources.webcam_capture",
    "ipcamera_capture": "framesource.sources.ipcamera_capture",
    "basler_capture": "framesource.sources.basler_capture",
    "genicam_capture": "framesource.sources.genicam_capture",
    "realsense_capture": "framesource.sources.realsense_capture",
    "ximea_capture": "framesource.sources.ximea_capture",
    "huateng_capture": "framesource.sources.huateng_capture",
    "video_file_capture": "framesource.sources.video_file_capture",
    "folder_capture": "framesource.sources.folder_capture",
    "screen_capture": "framesource.sources.screen_capture",
    "audiospectrogram_capture": "framesource.sources.audiospectrogram_capture",
}

for _alias, _target in _SUBMODULE_ALIASES.items():
    try:
        sys.modules[f"{__name__}.{_alias}"] = importlib.import_module(_target)
    except Exception:
        # Optional/vendor backend unavailable; skip aliasing it.
        pass
