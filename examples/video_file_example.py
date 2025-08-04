#!/usr/bin/env python3
"""
Video File Capture Example

Demonstrates video file playback with looping and real-time controls.
"""

import cv2
from frame_source import FrameSourceFactory


def main():
    """Test video file capture with playback controls."""
    cv2.namedWindow("Video File", cv2.WINDOW_NORMAL)
    print("Testing Video File Capture:")
    
    # Create video file capture (update path as needed)
    video_path = "../media/geti_demo.mp4"  # Adjust path relative to examples folder
    camera = FrameSourceFactory.create('video_file', source=video_path, loop=True)
    
    if not camera.connect():
        print(f"Failed to connect to video file: {video_path}")
        return
    
    if camera.is_connected:
        print(f"Video file: {video_path}")
        print(f"Frame size: {camera.get_frame_size()}")
        print(f"FPS: {camera.get_fps()}")
        print(f"Total frames: {getattr(camera, 'get_total_frames', lambda: 'Unknown')()}")
        
        def print_help():
            print("\nVideo File Controls:")
            print("  ESC or q - Quit")
            print("  h - Show this help")
            print("  SPACE - Pause/Resume")
            print("  r - Restart video")
        
        print_help()
        
        paused = False
        
        while camera.is_connected:
            if not paused:
                ret, frame = camera.read()
                if ret and frame is not None:
                    cv2.imshow("Video File", frame)
                elif not ret:
                    print("End of video reached")
                    if hasattr(camera, 'restart'):
                        camera.restart()
                        print("Restarting video...")
            
            key = cv2.waitKey(30) & 0xFF  # Longer wait for video playback
            if key == 27 or key == ord('q'):  # ESC or q to quit
                break
            elif key == ord('h'):
                print_help()
            elif key == ord(' '):  # Space to pause/resume
                paused = not paused
                print("Paused" if paused else "Resumed")
            elif key == ord('r'):  # Restart video
                if hasattr(camera, 'restart'):
                    camera.restart()
                    print("Restarting video...")
                    paused = False
    
    camera.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
