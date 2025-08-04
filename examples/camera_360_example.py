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
    print("Testing 360 Camera Capture:")
    
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
        fov=90,
        pitch=0.0,
        yaw=0.0,
        roll=0.0
    )
    camera.attach_processor(processor)
    
    camera.start_async()
    
    if camera.is_connected:
        print(f"Frame size: {camera.get_frame_size()}")
        print(f"FPS: {camera.get_fps()}")
        print(f"Processor FOV: {processor.get_parameter('fov')}°")
        
        def print_help():
            print("\n360 Camera Controls:")
            print("  ESC - Quit")
            print("  h - Show this help")
            print("  w/s - Adjust pitch (up/down)")
            print("  a/d - Adjust yaw (left/right)")
            print("  q/e - Adjust roll (left/right)")
            print("  r - Reset processor angles")
            print("  +/- - Adjust FOV")
        
        print_help()
        
        while camera.is_connected:
            ret, frame = camera.read()
            if ret and frame is not None:
                cv2.imshow("360 Camera", frame)
            
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
