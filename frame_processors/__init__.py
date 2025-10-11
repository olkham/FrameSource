"""Frame processors package for FrameSource.

This package contains various frame processing modules for handling different
types of image transformations and manipulations.
"""

from .frame_processor import FrameProcessor
from .equirectangular360_processor import Equirectangular2PinholeProcessor

# Conditionally import RealsenseDepthProcessor if pyrealsense2 is available
try:
    from .realsense_depth_processor import RealsenseDepthProcessor
    __all__ = [
        'FrameProcessor',
        'Equirectangular2PinholeProcessor',
        'RealsenseDepthProcessor'
    ]
except ImportError:
    # pyrealsense2 not available
    __all__ = [
        'FrameProcessor',
        'Equirectangular2PinholeProcessor',
    ]
