#!/usr/bin/env python3
"""
Multiple Cameras Example

Demonstrates connecting to multiple different camera types simultaneously
and viewing them in a grid layout.
"""

import cv2
from typing import List, Dict, Any
from frame_source import FrameSourceFactory


def main():
    """Test multiple cameras concurrently."""
    print("Testing Multiple Cameras:")
    
    # Configuration for different camera types
    # Uncomment/modify the cameras you want to test
    cameras_config: List[Dict[str, Any]] = [
        # {'capture_type': 'webcam', 'source': 0, 'threaded': True},
        # {'capture_type': 'basler', 'threaded': True},
        # {'capture_type': 'ximea', 'threaded': True},
        {'capture_type': 'ipcam', 'source': "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg", 'threaded': True},
        {'capture_type': 'video_file', 'source': "../media/geti_demo.mp4", 'loop': True, 'threaded': True},
        {'capture_type': 'folder', 'source': "../media/image_seq", 'sort_by': 'name', 'fps': 10, 'real_time': True, 'loop': True, 'threaded': True},
        {'capture_type': 'screen', 'x': 100, 'y': 100, 'w': 400, 'h': 300, 'fps': 15, 'threaded': True}
    ]
    
    capture_instances = []
    grid_cols = 2
    win_w, win_h = 640, 480
    
    # Connect to all cameras
    for idx, cam_cfg in enumerate(cameras_config):
        name = cam_cfg.pop('capture_type', None)
        if not name:
            print(f"Camera config missing 'capture_type': {cam_cfg}")
            continue
            
        print(f"Connecting to {name}...")
        
        try:
            camera = FrameSourceFactory.create(name, **cam_cfg)
            if camera.connect():
                # Configure window position in grid
                cv2.namedWindow(f"{name}", cv2.WINDOW_NORMAL)
                cv2.resizeWindow(f"{name}", win_w, win_h)
                col = idx % grid_cols
                row = idx // grid_cols
                x = col * (win_w + 10)
                y = row * (win_h + 50)  # Add space for title bar
                cv2.moveWindow(f"{name}", x, y)
                
                # Enable auto exposure if supported
                try:
                    camera.enable_auto_exposure(True)
                except:
                    pass  # Not all cameras support this
                
                # Start threaded capture
                camera.start_async()
                capture_instances.append((name, camera))
                print(f"✓ Connected to {name}")
            else:
                print(f"✗ Failed to connect to {name}")
        except Exception as e:
            print(f"✗ Error connecting to {name}: {e}")
    
    if not capture_instances:
        print("No cameras connected successfully")
        return
    
    print(f"\nConnected to {len(capture_instances)} cameras")
    print("Press 'q' to quit, 'h' for help")
    
    def print_help():
        print("\nMultiple Cameras Controls:")
        print("  q - Quit")
        print("  h - Show this help")
        print("  r - Reconnect all cameras")
    
    frame_counts = {name: 0 for name, _ in capture_instances}
    
    try:
        while True:
            active_cameras = 0
            
            for name, camera in capture_instances:
                if camera.is_connected:
                    ret, frame = camera.read()
                    if ret and frame is not None:
                        cv2.imshow(f"{name}", frame)
                        frame_counts[name] += 1
                        active_cameras += 1
                    else:
                        print(f"Failed to read frame from {name}")
                else:
                    print(f"{name} disconnected")
            
            if active_cameras == 0:
                print("All cameras disconnected")
                break
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                print_help()
            elif key == ord('r'):  # Reconnect all
                print("Reconnecting all cameras...")
                for name, camera in capture_instances:
                    if not camera.is_connected:
                        print(f"Reconnecting {name}...")
                        camera.disconnect()
                        if camera.connect():
                            camera.start_async()
                            print(f"✓ {name} reconnected")
                        else:
                            print(f"✗ {name} reconnection failed")
            
            # Print status every 500 frames for first camera
            if capture_instances and frame_counts[capture_instances[0][0]] % 500 == 0:
                status = ", ".join([f"{name}: {count}" for name, count in frame_counts.items()])
                print(f"Frame counts - {status}")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Clean up all cameras
        print("Disconnecting cameras...")
        for name, camera in capture_instances:
            try:
                if camera.is_connected:
                    camera.stop()
                camera.disconnect()
                print(f"✓ Disconnected {name}")
            except Exception as e:
                print(f"✗ Error disconnecting {name}: {e}")
        
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
