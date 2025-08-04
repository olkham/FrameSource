#!/usr/bin/env python3
"""
Screen Capture Example

Demonstrates live screen capture from a specific region of the desktop.
"""

import cv2
from frame_source import FrameSourceFactory


def main():
    """Test screen capture with region selection."""
    cv2.namedWindow("Screen Capture", cv2.WINDOW_NORMAL)
    print("Testing Screen Capture:")
    
    # Create screen capture for a specific region
    camera = FrameSourceFactory.create(
        'screen', 
        x=100, 
        y=100, 
        w=800, 
        h=600, 
        fps=30, 
        threaded=True
    )
    
    if not camera.connect():
        print("Failed to connect to screen capture")
        return
    
    camera.start_async()
    
    if camera.is_connected:
        print(f"Capture region: 100,100 800x600")
        print(f"Frame size: {camera.get_frame_size()}")
        print(f"FPS: {camera.get_fps()}")
        
        def print_help():
            print("\nScreen Capture Controls:")
            print("  ESC or q - Quit")
            print("  h - Show this help")
            print("  +/- - Increase/decrease FPS")
        
        print_help()
        
        while camera.is_connected:
            ret, frame = camera.read()
            if ret and frame is not None:
                cv2.imshow("Screen Capture", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q to quit
                break
            elif key == ord('h'):
                print_help()
            elif key == ord('+') or key == ord('='):  # Increase FPS
                current_fps = camera.get_fps()
                if current_fps:
                    new_fps = min(current_fps + 5, 60)
                    camera.set_fps(new_fps)
                    print(f"FPS: {new_fps}")
            elif key == ord('-'):  # Decrease FPS
                current_fps = camera.get_fps()
                if current_fps:
                    new_fps = max(current_fps - 5, 1)
                    camera.set_fps(new_fps)
                    print(f"FPS: {new_fps}")
    
    camera.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
