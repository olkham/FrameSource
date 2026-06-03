"""Deprecated compatibility shim for the old ``frame_processors`` package.

This package moved to :mod:`framesource.processors`. Importing
``frame_processors`` still works for now but emits a
:class:`DeprecationWarning` and will be removed in a future major release.
Update imports to ``framesource.processors``.
"""

import importlib
import sys
import warnings

warnings.warn(
    "'frame_processors' is deprecated and will be removed in a future major "
    "release; import from 'framesource.processors' instead.",
    DeprecationWarning,
    stacklevel=2,
)

import framesource.processors as _processors
from framesource.processors import *  # noqa: F401,F403

__all__ = getattr(_processors, "__all__", [])

_SUBMODULE_ALIASES = {
    "frame_processor": "framesource.processors.frame_processor",
    "equirectangular360_processor": "framesource.processors.equirectangular360_processor",
    "fisheye2equirectangular_processor": "framesource.processors.fisheye2equirectangular_processor",
    "realsense_depth_processor": "framesource.processors.realsense_depth_processor",
    "hyperspectral_processor": "framesource.processors.hyperspectral_processor",
}

for _alias, _target in _SUBMODULE_ALIASES.items():
    try:
        sys.modules[f"{__name__}.{_alias}"] = importlib.import_module(_target)
    except Exception:
        pass
