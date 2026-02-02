#!/usr/bin/env python3
"""
Fisheye PTZ Example

Demonstrates converting a 180-degree circular fisheye image to a controllable
pinhole camera view using the processor chain:
  Fisheye -> Equirectangular -> Pinhole

This enables pan/tilt/zoom control over a fisheye camera feed.
"""

import cv2
import numpy as np
from frame_source import FrameSourceFactory
from frame_processors import Fisheye2EquirectangularProcessor, Equirectangular2PinholeProcessor


def main():
    """Test fisheye to pinhole projection with PTZ controls."""
    cv2.namedWindow("Fisheye Source", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Equirectangular", cv2.WINDOW_NORMAL)
    cv2.namedWindow("PTZ View", cv2.WINDOW_NORMAL)

    print("Fisheye PTZ Example")
    print("=" * 50)

    # Global variables for mouse interaction
    pinhole_processor = None
    mouse_dragging = False

    def mouse_callback(event, x, y, flags, param):
        """Handle mouse interactions on the equirectangular image."""
        nonlocal pinhole_processor, mouse_dragging

        if pinhole_processor is None:
            return

        # Handle mouse wheel for FOV adjustment
        if event == cv2.EVENT_MOUSEWHEEL:
            current_fov = pinhole_processor.get_parameter('fov') or 90
            delta = flags
            fov_change = 5 if delta > 0 else -5
            new_fov = max(10, min(120, current_fov + fov_change))
            pinhole_processor.set_parameter('fov', new_fov)
            print(f"FOV: {new_fov}°")
            return

        # Handle mouse button events
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_dragging = True
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_dragging = False

        # Update angles when clicking or dragging
        if (event == cv2.EVENT_LBUTTONDOWN or
            (event == cv2.EVENT_MOUSEMOVE and mouse_dragging)):

            frame_width = param['frame_width']
            frame_height = param['frame_height']

            # Convert pixel coordinates to spherical coordinates
            longitude_deg, latitude_deg = pinhole_processor.pixel_to_spherical(
                x, y, frame_width, frame_height
            )

            # For fisheye, limit yaw to ±90° (front hemisphere)
            longitude_deg = max(-90, min(90, longitude_deg))

            current_roll = pinhole_processor.get_parameter('roll') or 0.0
            yaw, pitch, roll = pinhole_processor.spherical_to_processor_angles(
                longitude_deg, latitude_deg, current_roll
            )

            pinhole_processor.set_parameter('yaw', yaw)
            pinhole_processor.set_parameter('pitch', pitch)

            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"Look at: yaw={yaw:.1f}°, pitch={pitch:.1f}°")

    # Try to connect to a fisheye camera (webcam source 0)
    # For testing, you can also use a fisheye image/video file
    camera = FrameSourceFactory.create('webcam', source=0, threaded=True)

    if not camera.connect():
        print("No camera found. Creating a synthetic fisheye test pattern...")
        camera = None

        # Create a synthetic fisheye test image
        def create_synthetic_fisheye(size=800):
            """Create a synthetic circular fisheye test pattern."""
            img = np.zeros((size, size, 3), dtype=np.uint8)
            center = size // 2
            radius = size // 2 - 10

            # Draw concentric circles for elevation angles
            for i, r in enumerate(range(radius, 0, -radius // 6)):
                color = [(i * 40) % 255, (i * 60 + 100) % 255, (i * 80 + 50) % 255]
                cv2.circle(img, (center, center), r, color, 2)

            # Draw radial lines for azimuth angles
            for angle in range(0, 360, 30):
                rad = np.radians(angle)
                x2 = int(center + radius * np.cos(rad))
                y2 = int(center + radius * np.sin(rad))
                cv2.line(img, (center, center), (x2, y2), (100, 100, 100), 1)

            # Draw cardinal direction markers
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, "N", (center - 10, 40), font, 1, (255, 255, 255), 2)
            cv2.putText(img, "S", (center - 10, size - 20), font, 1, (255, 255, 255), 2)
            cv2.putText(img, "E", (size - 40, center + 10), font, 1, (255, 255, 255), 2)
            cv2.putText(img, "W", (20, center + 10), font, 1, (255, 255, 255), 2)

            # Draw center crosshair
            cv2.line(img, (center - 20, center), (center + 20, center), (0, 255, 0), 2)
            cv2.line(img, (center, center - 20), (center, center + 20), (0, 255, 0), 2)

            # Fill the outer area with black (simulating fisheye vignette)
            mask = np.zeros((size, size), dtype=np.uint8)
            cv2.circle(mask, (center, center), radius, 255, -1)
            img = cv2.bitwise_and(img, img, mask=mask)

            return img

        synthetic_frame = create_synthetic_fisheye(800)
    else:
        camera.start_async()
        print(f"Camera connected: {camera.get_frame_size()}")

    # Create fisheye to equirectangular processor
    fisheye_processor = Fisheye2EquirectangularProcessor(
        output_width=1920,
        output_height=960  # Standard 2:1 equirectangular aspect ratio
    )

    # Create equirectangular to pinhole processor for PTZ control
    pinhole_processor = Equirectangular2PinholeProcessor(
        output_width=1280,
        output_height=720,
        fov=90
    )

    # Initialize view angles
    pinhole_processor.set_parameter('pitch', 0.0)
    pinhole_processor.set_parameter('yaw', 0.0)
    pinhole_processor.set_parameter('roll', 0.0)

    # Set up mouse callback
    mouse_params = {'frame_width': 1920, 'frame_height': 960}
    cv2.setMouseCallback("Equirectangular", mouse_callback, mouse_params)

    def print_help():
        print("\nFisheye PTZ Controls:")
        print("  ESC - Quit")
        print("  h - Show this help")
        print("  LEFT CLICK on Equirectangular - Look in that direction")
        print("  DRAG - Hold and drag to pan view")
        print("  MOUSE WHEEL - Adjust FOV (zoom)")
        print("  w/s - Tilt up/down")
        print("  a/d - Pan left/right")
        print("  q/e - Roll left/right")
        print("  r - Reset view")
        print("  +/- - Adjust FOV")
        print("\nNote: Yaw is limited to ±90° (front hemisphere)")

    print_help()

    running = True
    while running:
        # Get frame from camera or use synthetic
        if camera is not None:
            ret, fisheye_frame = camera.read()
            if not ret or fisheye_frame is None:
                continue
        else:
            fisheye_frame = synthetic_frame.copy()

        # Process: Fisheye -> Equirectangular
        equi_frame = fisheye_processor.process(fisheye_frame)

        # Process: Equirectangular -> Pinhole (PTZ output)
        ptz_frame = pinhole_processor.process(equi_frame)

        # Update mouse params with actual equirectangular dimensions
        mouse_params['frame_width'] = equi_frame.shape[1]
        mouse_params['frame_height'] = equi_frame.shape[0]

        # Draw current view direction on equirectangular
        yaw = pinhole_processor.get_parameter('yaw') or 0
        pitch = pinhole_processor.get_parameter('pitch') or 0
        roll = pinhole_processor.get_parameter('roll') or 0

        eq_x, eq_y = pinhole_processor.processor_angles_to_equirectangular_coords(
            yaw, pitch, roll, equi_frame.shape[1], equi_frame.shape[0]
        )

        equi_display = equi_frame.copy()
        cv2.circle(equi_display, (eq_x, eq_y), 10, (0, 255, 0), 2)
        cv2.line(equi_display, (eq_x - 15, eq_y), (eq_x + 15, eq_y), (0, 255, 0), 2)
        cv2.line(equi_display, (eq_x, eq_y - 15), (eq_x, eq_y + 15), (0, 255, 0), 2)

        # Draw center crosshair on PTZ view
        cx, cy = ptz_frame.shape[1] // 2, ptz_frame.shape[0] // 2
        cv2.line(ptz_frame, (cx - 20, cy), (cx + 20, cy), (0, 255, 255), 2)
        cv2.line(ptz_frame, (cx, cy - 20), (cx, cy + 20), (0, 255, 255), 2)

        # Display all views
        cv2.imshow("Fisheye Source", fisheye_frame)
        cv2.imshow("Equirectangular", equi_display)
        cv2.imshow("PTZ View", ptz_frame)

        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            running = False
        elif key == ord('h'):
            print_help()
        elif key == ord('w'):
            current_pitch = pinhole_processor.get_parameter('pitch') or 0
            pinhole_processor.set_parameter('pitch', current_pitch + 5.0)
            print(f"Pitch: {pinhole_processor.get_parameter('pitch'):.1f}°")
        elif key == ord('s'):
            current_pitch = pinhole_processor.get_parameter('pitch') or 0
            pinhole_processor.set_parameter('pitch', current_pitch - 5.0)
            print(f"Pitch: {pinhole_processor.get_parameter('pitch'):.1f}°")
        elif key == ord('a'):
            current_yaw = pinhole_processor.get_parameter('yaw') or 0
            new_yaw = max(-90, current_yaw - 5.0)
            pinhole_processor.set_parameter('yaw', new_yaw)
            print(f"Yaw: {new_yaw:.1f}°")
        elif key == ord('d'):
            current_yaw = pinhole_processor.get_parameter('yaw') or 0
            new_yaw = min(90, current_yaw + 5.0)
            pinhole_processor.set_parameter('yaw', new_yaw)
            print(f"Yaw: {new_yaw:.1f}°")
        elif key == ord('q'):
            current_roll = pinhole_processor.get_parameter('roll') or 0
            pinhole_processor.set_parameter('roll', current_roll - 5.0)
            print(f"Roll: {pinhole_processor.get_parameter('roll'):.1f}°")
        elif key == ord('e'):
            current_roll = pinhole_processor.get_parameter('roll') or 0
            pinhole_processor.set_parameter('roll', current_roll + 5.0)
            print(f"Roll: {pinhole_processor.get_parameter('roll'):.1f}°")
        elif key == ord('r'):
            pinhole_processor.set_parameter('pitch', 0.0)
            pinhole_processor.set_parameter('yaw', 0.0)
            pinhole_processor.set_parameter('roll', 0.0)
            print("View reset")
        elif key == ord('+') or key == ord('='):
            current_fov = pinhole_processor.get_parameter('fov') or 90
            new_fov = min(current_fov + 5, 120)
            pinhole_processor.set_parameter('fov', new_fov)
            print(f"FOV: {new_fov}°")
        elif key == ord('-'):
            current_fov = pinhole_processor.get_parameter('fov') or 90
            new_fov = max(current_fov - 5, 10)
            pinhole_processor.set_parameter('fov', new_fov)
            print(f"FOV: {new_fov}°")

    # Cleanup
    if camera is not None:
        camera.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
