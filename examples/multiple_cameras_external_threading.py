#!/usr/bin/env python3
"""
Multiple Cameras Example with External Threading

Demonstrates connecting to multiple different camera types simultaneously
using external threading - eliminates timing issues and race conditions.
"""

import cv2
import time
import threading
import queue
from typing import List, Dict, Any
from framesource import FrameSourceFactory
from framesource.threading_utils import simple_frame_producer


def main():
    """Test multiple cameras with external threading."""
    print("Testing Multiple Cameras with External Threading:")
    
    # Configuration for different camera types
    cameras_config: List[Dict[str, Any]] = [
        {'capture_type': 'webcam', 'source': 0},
        # {'capture_type': 'basler'},
        # {'capture_type': 'basler'},
        # {'capture_type': 'ipcam', 'source': "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg"},
        # {'capture_type': 'video_file', 'source': "media/geti_demo.mp4", 'loop': True},
        # {'capture_type': 'folder', 'source': "media/image_seq", 'sort_by': 'name', 'fps': 10, 'real_time': True, 'loop': True},
        # {'capture_type': 'screen', 'x': 100, 'y': 100, 'w': 400, 'h': 300, 'fps': 15},
        # {'capture_type': 'realsense', 'width': 1280, 'height': 720},
    ]
    
    camera_systems = []  # List of (name, camera, queue, stop_event, thread)
    grid_cols = 2
    win_w, win_h = 640, 480
    
    # Connect to all cameras and start their producers
    for idx, cam_cfg in enumerate(cameras_config):
        name = cam_cfg.pop('capture_type', None)
        if not name:
            print(f"Camera config missing 'capture_type': {cam_cfg}")
            continue
            
        print(f"Setting up {name}...")
        
        try:
            # Create camera (synchronous only)
            camera = FrameSourceFactory.create(name, **cam_cfg)
            
            if camera.connect():
                # Configure window position in grid
                cv2.namedWindow(f"{name}", cv2.WINDOW_NORMAL)
                cv2.resizeWindow(f"{name}", win_w, win_h)
                col = idx % grid_cols
                row = idx // grid_cols
                x = col * (win_w + 10)
                y = row * (win_h + 50)
                cv2.moveWindow(f"{name}", x, y)
                
                # Enable auto exposure if supported
                try:
                    camera.enable_auto_exposure(True)
                except:
                    pass
                
                # Set up external threading
                frame_queue = queue.Queue(maxsize=5)
                stop_event = threading.Event()
                
                # Start producer thread
                producer_thread = threading.Thread(
                    target=simple_frame_producer,
                    args=(camera, frame_queue, stop_event, 30),  # 30 FPS target
                    daemon=True
                )
                producer_thread.start()
                
                camera_systems.append((name, camera, frame_queue, stop_event, producer_thread))
                print(f"✓ {name} producer started")
            else:
                print(f"✗ Failed to connect to {name}")
        except Exception as e:
            print(f"✗ Error setting up {name}: {e}")
    
    if not camera_systems:
        print("No cameras set up successfully")
        return
    
    print(f"\nSet up {len(camera_systems)} camera producers")
    print("Press 'q' to quit, 'h' for help")
    
    def print_help():
        print("\nMultiple Cameras Controls:")
        print("  q - Quit")
        print("  h - Show this help")
        print("  s - Show statistics")
    
    frame_counts = {name: 0 for name, _, _, _, _ in camera_systems}
    last_status_time = time.time()
    
    try:
        while True:
            active_cameras = 0
            
            for name, camera, frame_queue, stop_event, thread in camera_systems:
                if not stop_event.is_set() and thread.is_alive():
                    try:
                        # Get frame from queue (no timing issues!)
                        success, frame = frame_queue.get(timeout=0.05)
                        
                        if success and frame is not None:
                            cv2.imshow(f"{name}", frame)
                            frame_counts[name] += 1
                            active_cameras += 1
                        
                    except queue.Empty:
                        # No frame available yet - that's fine
                        active_cameras += 1  # Still count as active
                else:
                    print(f"{name} producer stopped")
            
            if active_cameras == 0:
                print("All camera producers stopped")
                break
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('h'):
                print_help()
            elif key == ord('s'):
                # Show statistics
                status = ", ".join([f"{name}: {count}" for name, count in frame_counts.items()])
                print(f"Frame counts - {status}")
            
            # Print status every 5 seconds
            current_time = time.time()
            if current_time - last_status_time > 5.0:
                status = ", ".join([f"{name}: {count}" for name, count in frame_counts.items()])
                print(f"Frame counts - {status}")
                last_status_time = current_time
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Clean shutdown - no race conditions!
        print("Stopping camera producers...")
        for name, camera, frame_queue, stop_event, thread in camera_systems:
            try:
                print(f"Stopping {name}...")
                stop_event.set()
                thread.join(timeout=2)
                print(f"✓ {name} stopped cleanly")
            except Exception as e:
                print(f"✗ Error stopping {name}: {e}")
        
        cv2.destroyAllWindows()
        
        # Final statistics
        total_frames = sum(frame_counts.values())
        print(f"\nFinal statistics:")
        for name, count in frame_counts.items():
            print(f"  {name}: {count} frames")
        print(f"  Total: {total_frames} frames")


if __name__ == "__main__":
    main()
