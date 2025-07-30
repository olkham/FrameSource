from enum import Enum
from typing import Dict, Any

import cv2
import numpy as np
import pyrealsense2 as rs
from .frame_processor import FrameProcessor, FrameType

class RealsenseProcessingOutput(Enum):
    RGB = 1
    ALIGNED_SIDE_BY_SIDE = 2
    ALIGNED_DEPTH = 4
    ALIGNED_DEPTH_COLORIZED = 5
    RGBD = 6

class RealsenseDepthProcessor(FrameProcessor):
    """Base class for all frame processors."""

    def __init__(self, output_format: RealsenseProcessingOutput = RealsenseProcessingOutput.ALIGNED_SIDE_BY_SIDE):
        super().__init__()
        self.output_format = output_format

    def process(self, frame: FrameType) -> FrameType:
        """Process a frame and return the processed frame."""
        if isinstance(frame, np.ndarray):
            return frame

        color_frame = frame['aligned_color']
        aligned_depth_frame = frame['aligned_depth']

        color_image = np.asanyarray(color_frame.get_data())
        aligned_depth_image = np.asanyarray(aligned_depth_frame.get_data())

        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(aligned_depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        if self.output_format == RealsenseProcessingOutput.RGB:
            return color_image
        elif self.output_format == RealsenseProcessingOutput.RGBD:
            rgbd = np.dstack((color_image, aligned_depth_image))
            return rgbd
        elif self.output_format == RealsenseProcessingOutput.ALIGNED_SIDE_BY_SIDE:
            # If depth and color resolutions are different, resize color image to match depth image for display
            if depth_colormap_dim != color_colormap_dim:
                resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                                 interpolation=cv2.INTER_AREA)
                images = np.hstack((resized_color_image, depth_colormap))
            else:
                images = np.hstack((color_image, depth_colormap))
            return images
        elif self.output_format == RealsenseProcessingOutput.ALIGNED_DEPTH:
            return aligned_depth_image
        elif self.output_format == RealsenseProcessingOutput.ALIGNED_DEPTH_COLORIZED:
            return depth_colormap

        return frame

    def set_parameter(self, name: str, value: Any) -> None:
        """Set a processing parameter."""
        self._parameters[name] = value

    def get_parameter(self, name: str) -> Any:
        """Get a processing parameter."""
        return self._parameters.get(name)

    def get_parameters(self) -> Dict[str, Any]:
        """Get all processing parameters."""
        return self._parameters.copy()
