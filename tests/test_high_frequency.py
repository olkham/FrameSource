#!/usr/bin/env python3
"""
Test script to verify high-frequency audio capture with AudioSpectrogramCapture.
Tests frequency ranges up to 20kHz to ensure proper sample rate adjustment.
"""

import cv2
import numpy as np
from frame_source import FrameSourceFactory

def test_high_frequency_audio():
    """Test audio spectrogram capture with high frequency range (20Hz - 20kHz)."""
    
    print("=== High Frequency Audio Test ===")
    print("Testing frequency range: 20Hz - 20kHz")
    
    # Create audio capture with high frequency range
    camera = FrameSourceFactory.create('audio_spectrogram',
                                      source=None,  # Default microphone
                                      freq_range=(20, 20000),  # Full audible range
                                      n_mels=256,  # More frequency bins for detail
                                      window_duration=2.0,
                                      frame_rate=30)  # Default grayscale colormap
    
    if not camera.connect():
        print("Failed to connect to audio source")
        return
    
    # Print configuration info
    print(f"Configuration:")
    if hasattr(camera, 'get_sample_rate'):
        print(f"  Sample rate: {camera.get_sample_rate()}Hz") # type: ignore
    if hasattr(camera, 'get_nyquist_frequency'):
        print(f"  Nyquist frequency: {camera.get_nyquist_frequency()}Hz") # type: ignore
    if hasattr(camera, 'get_freq_range'):
        print(f"  Frequency range: {camera.get_freq_range()}") # type: ignore
    if hasattr(camera, 'get_n_mels'):
        print(f"  Mel bands: {camera.get_n_mels()}") # type: ignore
    
    # Validate frequency range
    if hasattr(camera, 'validate_frequency_range'):
        is_valid, message = camera.validate_frequency_range(20, 20000) # type: ignore
        print(f"  Frequency range validation: {message}")
        if not is_valid:
            print("  WARNING: Frequency range validation failed!")
    
    # Start background threading
    camera.start()
    print("Started background audio capture thread")
    
    cv2.namedWindow("High Frequency Audio Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("High Frequency Audio Test", 800, 600)
    
    print("\nInstructions:")
    print("- Test with a tone generator from 20Hz to 20kHz")
    print("- You should see frequency content up to the Nyquist limit")
    print("- Press ESC to quit")
    print("- Press 'i' to print frequency info")
    
    try:
        while True:
            ret, frame = camera.read()
            
            if ret and frame is not None:
                # Add frequency scale overlay
                height, width = frame.shape[:2]
                
                # Add frequency markers (approximate)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                color = (255, 255, 255)
                thickness = 1
                
                # Add frequency labels on the right side
                freqs = [20, 100, 500, 1000, 2000, 5000, 10000, 20000]
                for freq in freqs:
                    if freq <= 20000:  # Only show frequencies within our range
                        # Calculate y position (mel scale is non-linear)
                        y_pos = int(height * (1 - np.log10(freq / 20) / np.log10(20000 / 20)))
                        y_pos = max(10, min(height - 10, y_pos))
                        
                        # Draw frequency label
                        if freq >= 1000:
                            label = f"{freq//1000}k"
                        else:
                            label = f"{freq}"
                        
                        cv2.putText(frame, label, (width - 40, y_pos), 
                                  font, font_scale, color, thickness)
                        cv2.line(frame, (width - 60, y_pos), (width - 45, y_pos), color, 1)
                
                cv2.imshow("High Frequency Audio Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord('i'):  # Info
                print("\nCurrent configuration:")
                if hasattr(camera, 'get_sample_rate'):
                    print(f"  Sample rate: {camera.get_sample_rate()}Hz") # type: ignore
                if hasattr(camera, 'get_nyquist_frequency'):
                    print(f"  Nyquist frequency: {camera.get_nyquist_frequency()}Hz") # type: ignore
                if hasattr(camera, 'get_freq_range'):
                    print(f"  Frequency range: {camera.get_freq_range()}") # type: ignore
                    
    finally:
        camera.stop()
        camera.disconnect()
        cv2.destroyAllWindows()
        print("Test completed")

if __name__ == "__main__":
    test_high_frequency_audio()
