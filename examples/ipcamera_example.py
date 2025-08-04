#!/usr/bin/env python3
"""
IP Camera Capture Example

Demonstrates IP camera capture using RTSP or HTTP streams.
"""

import cv2
from frame_source import FrameSourceFactory


def main():
    """Test IP camera capture."""
    cv2.namedWindow("IP Camera", cv2.WINDOW_NORMAL)
    print("Testing IP Camera Capture:")
    
    # Example IP camera URLs (update with your camera's URL)
    # rtsp_url = "rtsp://192.168.1.153:554/h264Preview_01_sub"
    # http_url = "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg"
    
    # Use a public demo camera for testing
    camera_url = "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg"
    
    camera = FrameSourceFactory.create(
        'ipcam', 
        source=camera_url,
        # username="admin",      # Uncomment if authentication needed
        # password="password",   # Uncomment if authentication needed
        threaded=True
    )
    
    if not camera.connect():
        print(f"Failed to connect to IP camera: {camera_url}")
        print("Make sure the URL is correct and the camera is accessible")
        return
    
    camera.start_async()
    
    if camera.is_connected:
        print(f"IP Camera URL: {camera_url}")
        print(f"Frame size: {camera.get_frame_size()}")
        print(f"FPS: {camera.get_fps()}")
        
        def print_help():
            print("\nIP Camera Controls:")
            print("  ESC or q - Quit")
            print("  h - Show this help")
            print("  r - Reconnect camera")
        
        print_help()
        
        frame_count = 0
        
        while camera.is_connected:
            ret, frame = camera.read()
            if ret and frame is not None:
                cv2.imshow("IP Camera", frame)
                frame_count += 1
                if frame_count % 100 == 0:  # Print status every 100 frames
                    print(f"Received {frame_count} frames")
            else:
                print("Failed to read frame from IP camera")
                # Try to reconnect after a short delay
                import time
                time.sleep(1)
                if not camera.is_connected:
                    print("Camera disconnected, attempting reconnect...")
                    camera.disconnect()
                    if camera.connect():
                        camera.start_async()
                        print("Reconnected successfully")
                    else:
                        print("Reconnection failed")
                        break
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q to quit
                break
            elif key == ord('h'):
                print_help()
            elif key == ord('r'):  # Reconnect
                print("Reconnecting to IP camera...")
                camera.disconnect()
                if camera.connect():
                    camera.start_async()
                    print("Reconnected successfully")
                else:
                    print("Reconnection failed")
    
    camera.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
