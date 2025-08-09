#!/usr/bin/env python3
"""
Simple test demonstrating external threading for FrameSource library.

This script shows how moving threading outside the core classes provides
better control and eliminates race conditions.
"""

import time
import threading
import queue
import cv2
import numpy as np
from typing import Optional, Tuple
from frame_source.webcam_capture import WebcamCapture
from frame_source.realsense_capture import RealsenseCapture
from frame_source.threading_utils import simple_frame_producer

n_frames = 1000


def test_current_vs_external():
    """Compare current implementation vs external threading."""
    
    print("=== Comparison: Current vs External Threading ===\n")
    
    # Test 1: Current implementation
    print("1. Testing CURRENT implementation (built-in threading):")
    camera_current = WebcamCapture(width=640, height=480)
    # camera_current = RealsenseCapture(width=640, height=480)
    
    camera_current.connect()
    # camera_current.set_frame_size(1920, 1080)
    # camera_current.set_fps(30)
    camera_current.start_async()  # Uses internal threading
    
    time.sleep(0.5)  # Allow connection to stabilize
    if camera_current.is_connected:
        print("   Connected successfully")
        frames_received = 0
        start_time = time.time()
        
        while frames_received < n_frames:
            success, frame = camera_current.read()
            if success and frame is not None:
                frames_received += 1
                cv2.imshow("Current Camera", frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
            # time.sleep(0.02)  # Simulate processing
        
        elapsed = time.time() - start_time
        print(f"   Received {frames_received} frames in {elapsed:.2f}s")
        print(f"   Average FPS: {frames_received/elapsed:.1f}")
        
        camera_current.stop()
        camera_current.disconnect()
    else:
        print("   Failed to connect to webcam")
    
    cv2.destroyAllWindows()
    print()
    
    # Test 2: External threading approach
    print("2. Testing EXTERNAL threading approach:")
    
    # Create synchronous capture source
    camera_external = WebcamCapture(width=640, height=480)
    # camera_external = RealsenseCapture(width=640, height=480)
    
    # camera_external.set_frame_size(1920, 1080)
    # camera_external.set_fps(30)
    
    # Set up external threading
    frame_queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    
    # Start producer thread
    producer_thread = threading.Thread(
        target=simple_frame_producer,
        args=(camera_external, frame_queue, stop_event),
        daemon=True
    )
    producer_thread.start()
    
    time.sleep(0.5)  # Let producer start
    
    frames_received = 0
    start_time = time.time()
    
    # Consumer loop
    while frames_received < n_frames:
        try:
            success, frame = frame_queue.get(timeout=0.1)
            if success and frame is not None:
                frames_received += 1
                # Simulate processing
                cv2.imshow("External Camera", frame)
                key = cv2.waitKey(1)
                if key == ord('q'):
                    break
        except queue.Empty:
            continue
    
    elapsed = time.time() - start_time
    print(f"   Received {frames_received} frames in {elapsed:.2f}s")
    print(f"   Average FPS: {frames_received/elapsed:.1f}")
    
    # Clean up
    stop_event.set()
    producer_thread.join(timeout=2)


def main():
    """Run all the tests."""
    print("FrameSource External Threading Demonstration")
    print("=" * 50)
    
    try:
        test_current_vs_external()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
