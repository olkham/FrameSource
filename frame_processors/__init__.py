"""Frame processors package for FrameSource.

This package contains various frame processing modules for handling different
types of image transformations and manipulations.
"""

from .frame_processor import FrameProcessor
from .equirectangular360_processor import Equirectangular2PinholeProcessor
from .realsense_depth_processor import RealsenseDepthProcessor

__all__ = [
    'FrameProcessor',
    'Equirectangular2PinholeProcessor',
    'RealsenseDepthProcessor'
]
