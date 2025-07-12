#!/usr/bin/env python3
"""
Test script to verify grayscale default and colormap switching functionality.
"""

import cv2
import numpy as np
from frame_source import FrameSourceFactory

def test_grayscale_default():
    """Test that grayscale is the default and colormaps can be switched."""
    
    print("=== Grayscale Default Test ===")
    
    # Create audio capture (should default to grayscale)
    camera = FrameSourceFactory.create('audio_spectrogram',
                                      source=None,
                                      n_mels=64,  # Smaller for faster processing
                                      window_duration=1.0,
                                      frame_rate=10)  # Slower for testing
    
    if not camera.connect():
        print("Failed to connect to audio source")
        return
    
    # Verify default is grayscale
    if hasattr(camera, 'get_colormap'):
        default_colormap = camera.get_colormap()
        print(f"Default colormap: {default_colormap} (None = grayscale)")
        if default_colormap is None:
            print("✓ Grayscale is correctly set as default")
        else:
            print("✗ Expected grayscale (None) as default")
    
    camera.start()
    print("Started background audio capture")
    
    cv2.namedWindow("Grayscale Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Grayscale Test", 600, 400)
    
    colormap_names = {
        None: "Grayscale",
        cv2.COLORMAP_VIRIDIS: "Viridis",
        cv2.COLORMAP_PLASMA: "Plasma",
        cv2.COLORMAP_HOT: "Hot",
        cv2.COLORMAP_JET: "Jet"
    }
    
    colormaps = list(colormap_names.keys())
    current_colormap_idx = 0
    
    print("\nInstructions:")
    print("- Press SPACE to cycle through colormaps")
    print("- Press ESC to quit")
    print("- Make some noise to see the spectrogram")
    
    try:
        while True:
            ret, frame = camera.read()
            
            if ret and frame is not None:
                # Add colormap name to frame
                current_colormap = colormaps[current_colormap_idx]
                colormap_name = colormap_names[current_colormap]
                
                cv2.putText(frame, f"Colormap: {colormap_name}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow("Grayscale Test", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            elif key == ord(' '):  # SPACE - cycle colormaps
                current_colormap_idx = (current_colormap_idx + 1) % len(colormaps)
                new_colormap = colormaps[current_colormap_idx]
                
                if hasattr(camera, 'set_colormap'):
                    camera.set_colormap(new_colormap)
                    colormap_name = colormap_names[new_colormap]
                    print(f"Switched to: {colormap_name}")
                    
    finally:
        camera.stop()
        camera.disconnect()
        cv2.destroyAllWindows()
        print("Test completed")

if __name__ == "__main__":
    test_grayscale_default()
