#!/usr/bin/env python3
"""
Image Folder Capture Example

Demonstrates capturing frames from a folder of images with sorting options
and real-time playback controls.
"""

import cv2
from frame_source import FrameSourceFactory


def main():
    """Test image folder capture with playback controls."""
    cv2.namedWindow("Image Folder", cv2.WINDOW_NORMAL)
    print("Testing Image Folder Capture:")
    
    # Create folder capture (update path as needed)
    folder_path = "../media/image_seq"  # Adjust path relative to examples folder
    camera = FrameSourceFactory.create(
        'folder', 
        source=folder_path, 
        sort_by='name',  # or 'date'
        fps=30, 
        real_time=True, 
        loop=True
    )
    
    if not camera.connect():
        print(f"Failed to connect to image folder: {folder_path}")
        return
    
    if camera.is_connected:
        print(f"Image folder: {folder_path}")
        print(f"Frame size: {camera.get_frame_size()}")
        print(f"FPS: {camera.get_fps()}")
        print(f"Total images: {getattr(camera, 'get_total_frames', lambda: 'Unknown')()}")
        
        def print_help():
            print("\nImage Folder Controls:")
            print("  ESC or q - Quit")
            print("  h - Show this help")
            print("  SPACE - Pause/Resume")
            print("  r - Restart sequence")
            print("  +/- - Increase/decrease FPS")
        
        print_help()
        
        paused = False
        
        while camera.is_connected:
            if not paused:
                ret, frame = camera.read()
                if ret and frame is not None:
                    cv2.imshow("Image Folder", frame)
                elif not ret:
                    print("End of sequence reached")
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q to quit
                break
            elif key == ord('h'):
                print_help()
            elif key == ord(' '):  # Space to pause/resume
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('r'):  # Restart sequence
                if hasattr(camera, 'restart'):
                    camera.restart()
                    print("Restarting image sequence...")
                    paused = False
            elif key == ord('+') or key == ord('='):  # Increase FPS
                current_fps = camera.get_fps()
                new_fps = min(current_fps + 5, 120)
                camera.set_fps(new_fps)
                print(f"FPS: {new_fps}")
            elif key == ord('-'):  # Decrease FPS
                current_fps = camera.get_fps()
                new_fps = max(current_fps - 5, 1)
                camera.set_fps(new_fps)
                print(f"FPS: {new_fps}")
    
    camera.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
