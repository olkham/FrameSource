import math
from typing import Tuple
import cv2
import numpy as np
from numba import njit, prange

from .frame_processor import FrameProcessor

# Optimized functions using Numba for better performance
@njit(cache=True, fastmath=True)
def generate_mapping_jit(output_width, output_height, focal_length, cx, cy,
                        R00, R01, R02, R10, R11, R12, R20, R21, R22,
                        frame_height, frame_width):
    """JIT-compiled coordinate mapping generation for maximum speed"""
    
    # Pre-allocate output arrays
    pixel_x = np.empty((output_height, output_width), dtype=np.float32)
    pixel_y = np.empty((output_height, output_width), dtype=np.float32)
    
    # Pre-calculate constants
    inv_focal = 1.0 / focal_length
    inv_2pi = 1.0 / (2.0 * np.pi)
    inv_pi = 1.0 / np.pi
    half_pi = np.pi * 0.5
    frame_width_minus_1 = frame_width - 1
    frame_height_minus_1 = frame_height - 1
    
    # Process each pixel
    for j in range(output_height):
        y_norm = (j - cy) * inv_focal
        for i in range(output_width):
            x_norm = (i - cx) * inv_focal
            
            # Normalize direction vector
            norm_factor = 1.0 / math.sqrt(x_norm * x_norm + y_norm * y_norm + 1.0)
            x_unit = x_norm * norm_factor
            y_unit = y_norm * norm_factor
            z_unit = norm_factor
            
            # Apply rotation matrix
            x_rot = R00 * x_unit + R01 * y_unit + R02 * z_unit
            y_rot = R10 * x_unit + R11 * y_unit + R12 * z_unit
            z_rot = R20 * x_unit + R21 * y_unit + R22 * z_unit
            
            # Convert to spherical coordinates
            theta = math.atan2(x_rot, z_rot)  # Azimuth
            phi = math.asin(max(-1.0, min(1.0, y_rot)))  # Elevation (clamped)
            
            # Convert to pixel coordinates
            u = (theta + np.pi) * inv_2pi
            v = (phi + half_pi) * inv_pi
            
            pixel_x[j, i] = max(0.0, min(frame_width_minus_1, u * frame_width_minus_1))
            pixel_y[j, i] = max(0.0, min(frame_height_minus_1, v * frame_height_minus_1))
    
    return pixel_x, pixel_y

@njit(cache=True, fastmath=True, parallel=True)
def generate_mapping_jit_parallel(output_width, output_height, focal_length, cx, cy,
                                 R00, R01, R02, R10, R11, R12, R20, R21, R22,
                                 frame_height, frame_width):
    """Parallel JIT-compiled coordinate mapping for multi-core systems"""
    
    # Pre-allocate output arrays
    pixel_x = np.empty((output_height, output_width), dtype=np.float32)
    pixel_y = np.empty((output_height, output_width), dtype=np.float32)
    
    # Pre-calculate constants
    inv_focal = 1.0 / focal_length
    inv_2pi = 1.0 / (2.0 * np.pi)
    inv_pi = 1.0 / np.pi
    half_pi = np.pi * 0.5
    frame_width_minus_1 = frame_width - 1
    frame_height_minus_1 = frame_height - 1
    
    # Process rows in parallel using prange
    for j in prange(output_height):
        y_norm = (j - cy) * inv_focal
        for i in range(output_width):
            x_norm = (i - cx) * inv_focal
            
            # Normalize direction vector
            norm_factor = 1.0 / math.sqrt(x_norm * x_norm + y_norm * y_norm + 1.0)
            x_unit = x_norm * norm_factor
            y_unit = y_norm * norm_factor
            z_unit = norm_factor
            
            # Apply rotation matrix
            x_rot = R00 * x_unit + R01 * y_unit + R02 * z_unit
            y_rot = R10 * x_unit + R11 * y_unit + R12 * z_unit
            z_rot = R20 * x_unit + R21 * y_unit + R22 * z_unit
            
            # Convert to spherical coordinates
            theta = math.atan2(x_rot, z_rot)
            phi = math.asin(max(-1.0, min(1.0, y_rot)))
            
            # Convert to pixel coordinates
            u = (theta + np.pi) * inv_2pi
            v = (phi + half_pi) * inv_pi
            
            pixel_x[j, i] = max(0.0, min(frame_width_minus_1, u * frame_width_minus_1))
            pixel_y[j, i] = max(0.0, min(frame_height_minus_1, v * frame_height_minus_1))
    
    return pixel_x, pixel_y

# Advanced optimized functions for maximum performance
@njit(cache=True, fastmath=True, inline='always')
def generate_mapping_jit_ultra(output_width, output_height, focal_length, cx, cy,
                              R00, R01, R02, R10, R11, R12, R20, R21, R22,
                              frame_height, frame_width):
    """Ultra-optimized coordinate mapping with memory layout optimization"""
    
    # Pre-allocate output arrays with optimal memory layout
    pixel_x = np.empty((output_height, output_width), dtype=np.float32)
    pixel_y = np.empty((output_height, output_width), dtype=np.float32)
    
    # Pre-calculate all constants (more than before)
    inv_focal = 1.0 / focal_length
    inv_2pi = 0.15915494309189535  # 1/(2*pi) precomputed
    inv_pi = 0.3183098861837907    # 1/pi precomputed
    half_pi = 1.5707963267948966   # pi/2 precomputed
    frame_width_f = float(frame_width - 1)
    frame_height_f = float(frame_height - 1)
    
    # Process each pixel with optimized inner loop
    for j in range(output_height):
        y_norm = (j - cy) * inv_focal
        y_norm_sq = y_norm * y_norm
        for i in range(output_width):
            x_norm = (i - cx) * inv_focal
            
            # Optimized normalization using precomputed y_norm_sq
            norm_factor = 1.0 / math.sqrt(x_norm * x_norm + y_norm_sq + 1.0)
            x_unit = x_norm * norm_factor
            y_unit = y_norm * norm_factor
            z_unit = norm_factor
            
            # Apply rotation matrix (unrolled for speed)
            x_rot = R00 * x_unit + R01 * y_unit + R02 * z_unit
            y_rot = R10 * x_unit + R11 * y_unit + R12 * z_unit
            z_rot = R20 * x_unit + R21 * y_unit + R22 * z_unit
            
            # Optimized spherical coordinate calculation
            theta = math.atan2(x_rot, z_rot)
            phi = math.asin(max(-1.0, min(1.0, y_rot)))
            
            # Direct pixel coordinate calculation with precomputed constants
            u = (theta + math.pi) * inv_2pi
            v = (phi + half_pi) * inv_pi
            
            # Final pixel coordinates with bounds checking
            pixel_x[j, i] = max(0.0, min(frame_width_f, u * frame_width_f))
            pixel_y[j, i] = max(0.0, min(frame_height_f, v * frame_height_f))
    
    return pixel_x, pixel_y

@njit(cache=True, fastmath=True, parallel=True)
def generate_mapping_jit_ultra_parallel(output_width, output_height, focal_length, cx, cy,
                                       R00, R01, R02, R10, R11, R12, R20, R21, R22,
                                       frame_height, frame_width):
    """Ultra-optimized parallel coordinate mapping using a flattened approach for better Numba parallelization"""
    
    # Pre-allocate output arrays
    pixel_x = np.empty((output_height, output_width), dtype=np.float32)
    pixel_y = np.empty((output_height, output_width), dtype=np.float32)
    
    # Pre-calculate constants
    inv_focal = 1.0 / focal_length
    inv_2pi = 0.15915494309189535
    inv_pi = 0.3183098861837907
    half_pi = 1.5707963267948966
    frame_width_f = float(frame_width - 1)
    frame_height_f = float(frame_height - 1)
    pi = 3.141592653589793
    
    # Flatten to 1D for better parallelization
    total_pixels = output_height * output_width
    
    # Use prange for parallel processing of individual pixels
    for idx in prange(total_pixels):
        # Convert 1D index back to 2D coordinates
        j = idx // output_width
        i = idx % output_width
        
        # Same calculation as before
        x_norm = (i - cx) * inv_focal
        y_norm = (j - cy) * inv_focal
        
        # Fast normalization
        norm_factor = 1.0 / math.sqrt(x_norm * x_norm + y_norm * y_norm + 1.0)
        x_unit = x_norm * norm_factor
        y_unit = y_norm * norm_factor
        z_unit = norm_factor
        
        # Matrix multiply
        x_rot = R00 * x_unit + R01 * y_unit + R02 * z_unit
        y_rot = R10 * x_unit + R11 * y_unit + R12 * z_unit
        z_rot = R20 * x_unit + R21 * y_unit + R22 * z_unit
        
        # Spherical conversion
        theta = math.atan2(x_rot, z_rot)
        phi = math.asin(max(-1.0, min(1.0, y_rot)))
        
        # Pixel mapping
        u = (theta + pi) * inv_2pi
        v = (phi + half_pi) * inv_pi
        
        pixel_x[j, i] = max(0.0, min(frame_width_f, u * frame_width_f))
        pixel_y[j, i] = max(0.0, min(frame_height_f, v * frame_height_f))
    
    return pixel_x, pixel_y


class Equirectangular2PinholeProcessor(FrameProcessor):
    """Convert equirectangular 360 frames to pinhole projections."""
    
    def __init__(self, fov: float = 90.0, output_width: int = 1920, output_height: int = 1080):
        super().__init__()
        self._parameters = {
            'roll': 0.0,
            'pitch': 0.0,
            'yaw': 0.0,
            'fov': fov,
            'output_width': output_width,
            'output_height': output_height
        }
        
        # Coordinate mapping cache
        self._map_cache = {}
        self._cache_size_limit = 50
    
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Convert equirectangular frame to pinhole projection."""
        roll = self._parameters['roll']
        pitch = self._parameters['pitch']
        yaw = self._parameters['yaw']
        fov = self._parameters['fov']
        output_width = self._parameters['output_width']
        output_height = self._parameters['output_height']
        
        # Normalize angles for consistent caching
        norm_yaw, norm_pitch, norm_roll = self._normalize_angles(yaw, pitch, roll)
        
        # Create cache key for coordinate mapping
        cache_key = (norm_yaw, norm_pitch, norm_roll, fov, output_width, output_height, frame.shape[0], frame.shape[1])
        
        # Check cache for coordinate mapping
        if cache_key in self._map_cache:
            pixel_x, pixel_y = self._map_cache[cache_key]
        else:
            # Generate new mapping
            pixel_x, pixel_y = self._generate_coordinate_mapping(
                norm_yaw, norm_pitch, norm_roll, fov, output_width, output_height, frame.shape
            )
            
            # Cache management
            if len(self._map_cache) >= self._cache_size_limit:
                oldest_key = next(iter(self._map_cache))
                del self._map_cache[oldest_key]
            
            self._map_cache[cache_key] = (pixel_x, pixel_y)
        
        # Apply remapping
        return cv2.remap(frame, pixel_x, pixel_y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)
    
    def _generate_coordinate_mapping(self, yaw: float, pitch: float, roll: float, fov: float, 
                                   output_width: int, output_height: int, frame_shape: Tuple[int, ...]) -> Tuple[np.ndarray, np.ndarray]:
        """Generate coordinate mapping for equirectangular to pinhole projection."""
        # Convert to radians
        yaw_rad = math.radians(yaw)
        pitch_rad = math.radians(pitch)
        roll_rad = math.radians(roll)
        fov_rad = math.radians(fov)
        
        # Pre-calculate constants
        focal_length = output_width / (2 * math.tan(fov_rad / 2))
        cx = output_width * 0.5
        cy = output_height * 0.5
        
        # Create rotation matrix elements
        cos_r, sin_r = math.cos(roll_rad), math.sin(roll_rad)
        cos_p, sin_p = math.cos(pitch_rad), math.sin(pitch_rad)
        cos_y, sin_y = math.cos(yaw_rad), math.sin(yaw_rad)
        
        # Combined rotation matrix elements
        R00 = cos_y * cos_r + sin_y * sin_r * sin_p
        R01 = cos_y * (-sin_r) + sin_y * cos_r * sin_p
        R02 = sin_y * cos_p
        R10 = sin_r * cos_p
        R11 = cos_r * cos_p
        R12 = -sin_p
        R20 = -sin_y * cos_r + cos_y * sin_r * sin_p
        R21 = -sin_y * (-sin_r) + cos_y * cos_r * sin_p
        R22 = cos_y * cos_p
        
        # Choose appropriate JIT implementation based on output size
        total_pixels = output_width * output_height
        
        if total_pixels > 1000000:
            return generate_mapping_jit_ultra_parallel(
                output_width, output_height, focal_length, cx, cy,
                R00, R01, R02, R10, R11, R12, R20, R21, R22,
                frame_shape[0], frame_shape[1]
            )
        elif total_pixels > 500000:
            return generate_mapping_jit_parallel(
                output_width, output_height, focal_length, cx, cy,
                R00, R01, R02, R10, R11, R12, R20, R21, R22,
                frame_shape[0], frame_shape[1]
            )
        elif total_pixels > 200000:
            return generate_mapping_jit_ultra(
                output_width, output_height, focal_length, cx, cy,
                R00, R01, R02, R10, R11, R12, R20, R21, R22,
                frame_shape[0], frame_shape[1]
            )
        else:
            return generate_mapping_jit(
                output_width, output_height, focal_length, cx, cy,
                R00, R01, R02, R10, R11, R12, R20, R21, R22,
                frame_shape[0], frame_shape[1]
            )
    
    def _normalize_angles(self, yaw: float, pitch: float, roll: float) -> Tuple[float, float, float]:
        """Normalize angles to canonical ranges for cache consistency."""
        # Normalize yaw to [0, 360)
        yaw = yaw % 360
        
        # Normalize pitch properly
        pitch = ((pitch + 180) % 360) - 180
        
        # Handle pitch overflow beyond valid range [-90, 90]
        if pitch > 90:
            pitch = 180 - pitch
            yaw = (yaw + 180) % 360
            roll = (roll + 180) % 360
        elif pitch < -90:
            pitch = -180 - pitch
            yaw = (yaw + 180) % 360
            roll = (roll + 180) % 360
        
        # Normalize roll to [0, 360)
        roll = roll % 360
        
        return yaw, pitch, roll
    
    def clear_cache(self) -> None:
        """Clear the coordinate mapping cache."""
        self._map_cache.clear()
