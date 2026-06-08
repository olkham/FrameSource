import math
from typing import Tuple, Optional
import cv2
import numpy as np
from numba import njit, prange

from .frame_processor import FrameProcessor, FrameType


@njit(cache=True, fastmath=True)
def generate_fisheye_mapping_jit(output_width, output_height,
                                  fisheye_cx, fisheye_cy, fisheye_radius,
                                  frame_height, frame_width):
    """JIT-compiled coordinate mapping for fisheye to equirectangular conversion.

    Maps a 180-degree equidistant fisheye (circular) image to equirectangular format.
    The front hemisphere of the equirectangular output contains the fisheye content,
    while the back hemisphere (beyond ±90° yaw) will sample outside the fisheye circle.
    """

    pixel_x = np.empty((output_height, output_width), dtype=np.float32)
    pixel_y = np.empty((output_height, output_width), dtype=np.float32)

    # Pre-calculate constants
    pi = 3.141592653589793
    half_pi = pi * 0.5
    two_pi = pi * 2.0

    # For each output pixel, calculate corresponding fisheye coordinates
    for j in range(output_height):
        # Latitude (elevation): top of image is +90°, bottom is -90°
        # v ranges from 0 to 1
        v = j / (output_height - 1) if output_height > 1 else 0.5
        latitude = half_pi - v * pi  # +90° to -90°

        for i in range(output_width):
            # Longitude (azimuth): left edge is -180°, right edge is +180°
            # u ranges from 0 to 1
            u = i / (output_width - 1) if output_width > 1 else 0.5
            longitude = (u * two_pi) - pi  # -180° to +180°

            # Convert spherical coordinates to 3D unit vector
            # Convention: Z is forward, X is right, Y is up
            cos_lat = math.cos(latitude)
            x = cos_lat * math.sin(longitude)
            y = math.sin(latitude)
            z = cos_lat * math.cos(longitude)

            # For fisheye with optical axis along +Z:
            # Calculate angle from optical axis (z-axis)
            # z = cos(theta) where theta is angle from optical axis
            theta = math.acos(max(-1.0, min(1.0, z)))

            # Calculate azimuthal angle in the fisheye plane (XY plane)
            phi = math.atan2(y, x)

            # For equidistant fisheye projection:
            # r = (theta / (pi/2)) * fisheye_radius
            # This maps 0° (center) to r=0, and 90° (edge) to r=fisheye_radius
            # Beyond 90° (back hemisphere), r > fisheye_radius (will sample outside circle)
            r = (theta / half_pi) * fisheye_radius

            # Convert polar (r, phi) to cartesian fisheye pixel coordinates
            fisheye_x = fisheye_cx + r * math.cos(phi)
            fisheye_y = fisheye_cy - r * math.sin(phi)  # Negative because image Y increases downward

            pixel_x[j, i] = fisheye_x
            pixel_y[j, i] = fisheye_y

    return pixel_x, pixel_y


@njit(cache=True, fastmath=True, parallel=True)
def generate_fisheye_mapping_jit_parallel(output_width, output_height,
                                          fisheye_cx, fisheye_cy, fisheye_radius,
                                          frame_height, frame_width):
    """Parallel JIT-compiled coordinate mapping for fisheye to equirectangular conversion."""

    pixel_x = np.empty((output_height, output_width), dtype=np.float32)
    pixel_y = np.empty((output_height, output_width), dtype=np.float32)

    pi = 3.141592653589793
    half_pi = pi * 0.5
    two_pi = pi * 2.0

    total_pixels = output_height * output_width

    for idx in prange(total_pixels):
        j = idx // output_width
        i = idx % output_width

        v = j / (output_height - 1) if output_height > 1 else 0.5
        latitude = half_pi - v * pi

        u = i / (output_width - 1) if output_width > 1 else 0.5
        longitude = (u * two_pi) - pi

        cos_lat = math.cos(latitude)
        x = cos_lat * math.sin(longitude)
        y = math.sin(latitude)
        z = cos_lat * math.cos(longitude)

        theta = math.acos(max(-1.0, min(1.0, z)))
        phi = math.atan2(y, x)

        r = (theta / half_pi) * fisheye_radius

        fisheye_x = fisheye_cx + r * math.cos(phi)
        fisheye_y = fisheye_cy - r * math.sin(phi)

        pixel_x[j, i] = fisheye_x
        pixel_y[j, i] = fisheye_y

    return pixel_x, pixel_y


class Fisheye2EquirectangularProcessor(FrameProcessor):
    """Convert 180-degree circular fisheye images to equirectangular format.

    This processor takes a circular fisheye image (typically from a 180° fisheye lens)
    and projects it onto an equirectangular format. The output can then be chained
    with Equirectangular2PinholeProcessor to enable pan/tilt/zoom control.

    The fisheye is assumed to use equidistant projection, which is common for
    180-degree fisheye lenses. The circular fisheye image is centered and fills
    the front hemisphere (±90° from center) of the equirectangular output.

    Parameters:
        fisheye_cx: X coordinate of fisheye center in input image (default: auto-detect as image center)
        fisheye_cy: Y coordinate of fisheye center in input image (default: auto-detect as image center)
        fisheye_radius: Radius of the fisheye circle in pixels (default: auto-detect as min(width, height)/2)
        output_width: Width of equirectangular output (default: 1920)
        output_height: Height of equirectangular output (default: 960, standard 2:1 aspect ratio)

    Example usage:
        # Create processor chain for fisheye PTZ control
        fisheye2equi = Fisheye2EquirectangularProcessor(output_width=1920, output_height=960)
        equi2pinhole = Equirectangular2PinholeProcessor(fov=90, output_width=1280, output_height=720)

        # Process frame
        equi_frame = fisheye2equi.process(fisheye_frame)
        output_frame = equi2pinhole.process(equi_frame)

        # Control view direction
        equi2pinhole.set_parameter('yaw', 30)   # Pan right 30°
        equi2pinhole.set_parameter('pitch', -15) # Tilt up 15°
    """

    def __init__(self, output_width: int = 1920, output_height: int = 960,
                 fisheye_cx: Optional[float] = None, fisheye_cy: Optional[float] = None,
                 fisheye_radius: Optional[float] = None):
        super().__init__()
        self._parameters = {
            'fisheye_cx': fisheye_cx,
            'fisheye_cy': fisheye_cy,
            'fisheye_radius': fisheye_radius,
            'output_width': output_width,
            'output_height': output_height
        }

        # Coordinate mapping cache
        self._map_cache = {}
        self._cache_size_limit = 10

    def process(self, frame: FrameType) -> FrameType:
        """Convert fisheye frame to equirectangular projection."""
        output_width = self._parameters['output_width']
        output_height = self._parameters['output_height']

        # Auto-detect fisheye parameters if not specified
        frame_height, frame_width = frame.shape[:2]

        fisheye_cx = self._parameters['fisheye_cx']
        if fisheye_cx is None:
            fisheye_cx = frame_width / 2.0

        fisheye_cy = self._parameters['fisheye_cy']
        if fisheye_cy is None:
            fisheye_cy = frame_height / 2.0

        fisheye_radius = self._parameters['fisheye_radius']
        if fisheye_radius is None:
            fisheye_radius = min(frame_width, frame_height) / 2.0

        # Create cache key
        cache_key = (fisheye_cx, fisheye_cy, fisheye_radius,
                     output_width, output_height, frame_height, frame_width)

        # Check cache for coordinate mapping
        if cache_key in self._map_cache:
            pixel_x, pixel_y = self._map_cache[cache_key]
        else:
            # Generate new mapping
            pixel_x, pixel_y = self._generate_coordinate_mapping(
                fisheye_cx, fisheye_cy, fisheye_radius,
                output_width, output_height, frame_height, frame_width
            )

            # Cache management
            if len(self._map_cache) >= self._cache_size_limit:
                oldest_key = next(iter(self._map_cache))
                del self._map_cache[oldest_key]

            self._map_cache[cache_key] = (pixel_x, pixel_y)

        # Apply remapping with black border for out-of-bounds areas
        return cv2.remap(frame, pixel_x, pixel_y, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    def _generate_coordinate_mapping(self, fisheye_cx: float, fisheye_cy: float,
                                     fisheye_radius: float, output_width: int,
                                     output_height: int, frame_height: int,
                                     frame_width: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate coordinate mapping for fisheye to equirectangular projection."""

        total_pixels = output_width * output_height

        if total_pixels > 500000:
            return generate_fisheye_mapping_jit_parallel(
                output_width, output_height,
                fisheye_cx, fisheye_cy, fisheye_radius,
                frame_height, frame_width
            )
        else:
            return generate_fisheye_mapping_jit(
                output_width, output_height,
                fisheye_cx, fisheye_cy, fisheye_radius,
                frame_height, frame_width
            )

    def clear_cache(self) -> None:
        """Clear the coordinate mapping cache."""
        self._map_cache.clear()

    def set_fisheye_circle(self, cx: float, cy: float, radius: float) -> None:
        """Set the fisheye circle parameters.

        Args:
            cx: X coordinate of fisheye center
            cy: Y coordinate of fisheye center
            radius: Radius of the fisheye circle in pixels
        """
        self._parameters['fisheye_cx'] = cx
        self._parameters['fisheye_cy'] = cy
        self._parameters['fisheye_radius'] = radius
        self.clear_cache()

    def auto_detect_fisheye_circle(self, frame: FrameType) -> Tuple[float, float, float]:
        """Attempt to auto-detect the fisheye circle from the frame.

        This uses a simple heuristic based on finding the largest dark border.
        For best results, manually specify the fisheye parameters.

        Args:
            frame: Input fisheye frame

        Returns:
            Tuple of (cx, cy, radius)
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        # Threshold to find the bright fisheye area
        _, binary = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Find the largest contour
            largest = max(contours, key=cv2.contourArea)

            # Fit a circle to the contour
            (cx, cy), radius = cv2.minEnclosingCircle(largest)

            return float(cx), float(cy), float(radius)

        # Fallback to image center
        h, w = frame.shape[:2]
        return w / 2.0, h / 2.0, min(w, h) / 2.0
