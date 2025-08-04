#!/usr/bin/env python3
"""
360 Camera Example with Equirectangular Processing

Demonstrates 360° camera capture with equirectangular to pinhole projection.
Includes interactive controls for adjusting the virtual camera view.
"""

import cv2
from frame_source import FrameSourceFactory
from frame_processors.equirectangular360_processor import Equirectangular2PinholeProcessor



def main():
    """Test 360 camera with equirectangular to pinhole projection."""
    cv2.namedWindow("360 Camera", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Projected", cv2.WINDOW_NORMAL)

    print("Testing 360 Camera Capture:")
    
    # Global variables for mouse callback
    processor = None
    mouse_dragging = False
    
    def mouse_callback(event, x, y, flags, param):
        """Handle mouse interactions on the equirectangular image."""
        nonlocal processor, mouse_dragging
        
        if processor is None:
            return
            
        # Handle mouse wheel for FOV adjustment
        if event == cv2.EVENT_MOUSEWHEEL:
            current_fov = processor.get_parameter('fov') or 90
            # Mouse wheel delta is positive for scroll up, negative for scroll down
            delta = flags  # In OpenCV, flags contains the wheel delta for mouse wheel events
            fov_change = 5 if delta > 0 else -5
            new_fov = max(10, min(180, current_fov + fov_change))
            processor.set_parameter('fov', new_fov)
            print(f"Mouse wheel -> FOV: {new_fov}°")
            return
        
        # Handle mouse button events
        if event == cv2.EVENT_LBUTTONDOWN:
            mouse_dragging = True
        elif event == cv2.EVENT_LBUTTONUP:
            mouse_dragging = False
        
        # Update angles when clicking or dragging
        if (event == cv2.EVENT_LBUTTONDOWN or 
            (event == cv2.EVENT_MOUSEMOVE and mouse_dragging)):
            
            # Get image dimensions
            frame_width = param['frame_width']
            frame_height = param['frame_height']
            
            # Convert pixel coordinates to spherical coordinates using helper function
            longitude_deg, latitude_deg = processor.pixel_to_spherical(
                x, y, frame_width, frame_height
            )
            
            # Convert to processor angles using helper function
            current_roll = processor.get_parameter('roll') or 0.0
            yaw, pitch, roll = processor.spherical_to_processor_angles(
                longitude_deg, latitude_deg, current_roll
            )
            
            # Set processor angles
            processor.set_parameter('yaw', yaw)
            processor.set_parameter('pitch', pitch)
            processor.set_parameter('roll', roll)
            
            # Only print on initial click to avoid spam during dragging
            if event == cv2.EVENT_LBUTTONDOWN:
                print(f"Clicked at pixel ({x}, {y}) -> Set angles: yaw={yaw:.1f}°, pitch={pitch:.1f}°, roll={roll:.1f}°")
    
    # Create webcam capture for 360 camera (adjust source as needed)
    camera = FrameSourceFactory.create('webcam', source=0, threaded=True)
    
    if not camera.connect():
        print("Failed to connect to 360 camera")
        return
    
    # Set camera resolution for Insta360 X5 webcam mode
    camera.set_frame_size(2880, 1440)
    camera.set_fps(30)
    
    # Create and attach equirectangular processor
    processor = Equirectangular2PinholeProcessor(
        output_width=1920,
        output_height=1080,
        fov=90
    )
    
    # Set initial viewing angles (these are parameters, not constructor args)
    processor.set_parameter('pitch', 0.0)
    processor.set_parameter('yaw', 0.0)
    processor.set_parameter('roll', 0.0)
    
    # Set up mouse callback for interactive clicking
    mouse_params = {'frame_width': 2880, 'frame_height': 1440}
    cv2.setMouseCallback("360 Camera", mouse_callback, mouse_params)
    
    # camera.attach_processor(processor)
    
    camera.start_async()
    
    if camera.is_connected:
        print(f"Frame size: {camera.get_frame_size()}")
        print(f"FPS: {camera.get_fps()}")
        print(f"Processor FOV: {processor.get_parameter('fov')}°")
        
        def print_help():
            print("\n360 Camera Controls:")
            print("  ESC - Quit")
            print("  h - Show this help")
            print("  LEFT CLICK - Click anywhere on 360° image to look in that direction")
            print("  DRAG - Hold left mouse button and drag to continuously pan view")
            print("  MOUSE WHEEL - Scroll to adjust FOV (field of view)")
            print("  w/s - Adjust pitch (up/down)")
            print("  a/d - Adjust yaw (left/right)")
            print("  q/e - Adjust roll (left/right)")
            print("  r - Reset processor angles")
            print("  +/- - Adjust FOV")
        
        print_help()
        
        while camera.is_connected:
            ret, frame = camera.read()
            # if ret and frame is not None:
                # cv2.imshow("360 Camera", frame)
            
            if frame is not None:
                # Update mouse callback parameters with actual frame dimensions
                mouse_params['frame_width'] = frame.shape[1]
                mouse_params['frame_height'] = frame.shape[0]
                
                # Create a copy for display to avoid modifying the original frame
                display_frame = frame.copy()
                projected = processor.process(frame)  # Process original frame
                
                # Ensure projected is a numpy array (not dict)
                if isinstance(projected, dict):
                    continue  # Skip this frame if processor returns unexpected format
                
                # Back-project center of projected image to equirectangular coordinates using static helper
                center_x_proj = projected.shape[1] // 2
                center_y_proj = projected.shape[0] // 2

                # Get current processor angles
                pitch = processor.get_parameter('pitch') or 0
                yaw = processor.get_parameter('yaw') or 0
                roll = processor.get_parameter('roll') or 0

                # Convert processor angles to equirectangular coordinates using helper function
                eq_x, eq_y = processor.processor_angles_to_equirectangular_coords(
                    yaw, pitch, roll, frame.shape[1], frame.shape[0]
                )

                # Draw crosshair on DISPLAY COPY of equirectangular frame (green)
                cv2.line(display_frame, (eq_x - 15, eq_y), (eq_x + 15, eq_y), (0, 255, 0), 3)
                cv2.line(display_frame, (eq_x, eq_y - 15), (eq_x, eq_y + 15), (0, 255, 0), 3)

                # Draw crosshair on projected frame center (yellow)
                cv2.line(projected, (center_x_proj - 15, center_y_proj), (center_x_proj + 15, center_y_proj), (0, 255, 255), 3)
                cv2.line(projected, (center_x_proj, center_y_proj - 15), (center_x_proj, center_y_proj + 15), (0, 255, 255), 3)

                # Display both frames
                cv2.imshow("360 Camera", display_frame)
                cv2.imshow("Projected", projected)

                # Only print coordinates occasionally to reduce flickering
                frame_count = getattr(main, 'frame_count', 0)
                main.frame_count = frame_count + 1
                if frame_count % 30 == 0:  # Print every 30 frames (~1 second at 30fps)
                    print(f"Center projects to equirectangular coords: ({eq_x}, {eq_y}) | Angles: pitch={pitch:.1f}°, yaw={yaw:.1f}°")
                
                

            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to quit
                break
            elif key == ord('h'):  # Show help
                print_help()
            elif key == ord('w'):  # Pitch up
                current_pitch = processor.get_parameter('pitch') or 0
                processor.set_parameter('pitch', current_pitch + 5.0)
                print(f"Pitch: {processor.get_parameter('pitch'):.1f}°")
            elif key == ord('s'):  # Pitch down
                current_pitch = processor.get_parameter('pitch') or 0
                processor.set_parameter('pitch', current_pitch - 5.0)
                print(f"Pitch: {processor.get_parameter('pitch'):.1f}°")
            elif key == ord('a'):  # Yaw left
                current_yaw = processor.get_parameter('yaw') or 0
                processor.set_parameter('yaw', current_yaw - 5.0)
                print(f"Yaw: {processor.get_parameter('yaw'):.1f}°")
            elif key == ord('d'):  # Yaw right
                current_yaw = processor.get_parameter('yaw') or 0
                processor.set_parameter('yaw', current_yaw + 5.0)
                print(f"Yaw: {processor.get_parameter('yaw'):.1f}°")
            elif key == ord('q'):  # Roll left
                current_roll = processor.get_parameter('roll') or 0
                processor.set_parameter('roll', current_roll - 5.0)
                print(f"Roll: {processor.get_parameter('roll'):.1f}°")
            elif key == ord('e'):  # Roll right
                current_roll = processor.get_parameter('roll') or 0
                processor.set_parameter('roll', current_roll + 5.0)
                print(f"Roll: {processor.get_parameter('roll'):.1f}°")
            elif key == ord('r'):  # Reset processor angles
                processor.set_parameter('pitch', 0.0)
                processor.set_parameter('yaw', 0.0)
                processor.set_parameter('roll', 0.0)
                print("Processor angles reset to 0°")
            elif key == ord('+') or key == ord('='):  # Increase FOV
                current_fov = processor.get_parameter('fov') or 90
                new_fov = min(current_fov + 5, 180)
                processor.set_parameter('fov', new_fov)
                print(f"FOV: {new_fov}°")
            elif key == ord('-'):  # Decrease FOV
                current_fov = processor.get_parameter('fov') or 90
                new_fov = max(current_fov - 5, 10)
                processor.set_parameter('fov', new_fov)
                print(f"FOV: {new_fov}°")
    
    camera.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
