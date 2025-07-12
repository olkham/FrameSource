#!/usr/bin/env python3
"""
Test script to demonstrate background threading support in AudioSpectrogramCapture.
Compares performance between threaded and non-threaded modes.
"""

import time
import cv2
from frame_source import FrameSourceFactory

def test_threading_performance():
    """Compare threaded vs non-threaded performance for audio capture."""
    
    print("=== Audio Spectrogram Threading Test ===\n")
    
    # Test both modes
    for threaded_mode in [False, True]:
        mode_name = "Threaded" if threaded_mode else "Non-threaded"
        print(f"Testing {mode_name} mode...")
        
        # Create audio capture
        camera = FrameSourceFactory.create('audio_spectrogram', 
                                          source=None,  # Default microphone
                                          n_mels=64,    # Smaller for faster processing
                                          frame_rate=30,
                                          window_duration=1.0)
        
        if not camera.connect():
            print(f"Failed to connect to audio source for {mode_name} mode")
            continue
            
        # Start threading if requested
        if threaded_mode:
            camera.start()
            print("  Background thread started")
            
        # Measure frame capture performance
        num_frames = 60
        start_time = time.time()
        successful_frames = 0
        
        for i in range(num_frames):
            ret, frame = camera.read()
            if ret and frame is not None:
                successful_frames += 1
                
            # Small delay to simulate real-world usage
            time.sleep(0.01)
            
        end_time = time.time()
        
        # Calculate performance metrics
        total_time = end_time - start_time
        fps = successful_frames / total_time if total_time > 0 else 0
        
        print(f"  Results for {mode_name} mode:")
        print(f"    Frames captured: {successful_frames}/{num_frames}")
        print(f"    Time taken: {total_time:.2f} seconds")
        print(f"    Average FPS: {fps:.2f}")
        
        # Cleanup
        if threaded_mode:
            camera.stop()
            print("  Background thread stopped")
            
        camera.disconnect()
        print(f"  {mode_name} mode test completed\n")

def test_interactive_audio():
    """Interactive test showing real-time audio spectrogram with threading."""
    
    print("=== Interactive Audio Spectrogram Test ===")
    print("This will show a real-time audio spectrogram using background threading.")
    print("Press ESC to quit, SPACE to toggle between threaded/non-threaded modes\n")
    
    cv2.namedWindow("Audio Spectrogram - Threading Test", cv2.WINDOW_NORMAL)
    
    camera = FrameSourceFactory.create('audio_spectrogram',
                                      source=None,  # Default microphone 
                                      n_mels=128,
                                      frame_rate=30,
                                      window_duration=2.0,
                                      colormap=cv2.COLORMAP_VIRIDIS)
    
    if not camera.connect():
        print("Failed to connect to audio source")
        return
        
    # Start in threaded mode
    threaded = True
    camera.start()
    print("Started in threaded mode")
    
    frame_count = 0
    fps_start_time = time.time()
    
    try:
        while True:
            ret, frame = camera.read()
            
            if ret and frame is not None:
                frame_count += 1
                
                # Calculate and display FPS
                if frame_count % 30 == 0:
                    current_time = time.time()
                    elapsed = current_time - fps_start_time
                    fps = 30 / elapsed if elapsed > 0 else 0
                    mode = "Threaded" if threaded else "Non-threaded"
                    print(f"{mode} mode - FPS: {fps:.1f}")
                    fps_start_time = current_time
                
                # Add mode text to frame
                mode_text = "THREADED" if threaded else "NON-THREADED"
                cv2.putText(frame, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                           1, (255, 255, 255), 2)
                
                cv2.imshow("Audio Spectrogram - Threading Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE - toggle threading mode
                if threaded:
                    camera.stop()
                    threaded = False
                    print("Switched to non-threaded mode")
                else:
                    camera.start()
                    threaded = True
                    print("Switched to threaded mode")
                frame_count = 0
                fps_start_time = time.time()
                
    finally:
        if threaded:
            camera.stop()
        camera.disconnect()
        cv2.destroyAllWindows()
        print("Interactive test completed")

if __name__ == "__main__":
    try:
        # Run performance comparison first
        test_threading_performance()
        
        # Ask user if they want to run interactive test
        response = input("Run interactive audio spectrogram test? (y/n): ")
        if response.lower().startswith('y'):
            test_interactive_audio()
            
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during testing: {e}")
