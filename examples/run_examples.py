#!/usr/bin/env python3
"""
FrameSource Examples

This directory contains organized examples for all FrameSource capture types.
Run individual examples or use this script to choose which example to run.
"""

import os
import sys
import subprocess


def main():
    """Interactive example selector."""
    examples = {
        '1': ('webcam_example.py', 'Webcam Capture - Basic webcam with exposure/gain controls'),
        '2': ('audio_spectrogram_example.py', 'Audio Spectrogram - Microphone to visual spectrogram'),
        '3': ('camera_360_example.py', '360° Camera - Equirectangular to pinhole projection'),
        '4': ('video_file_example.py', 'Video File - Playback with controls'),
        '5': ('image_folder_example.py', 'Image Folder - Sequence playback'),
        '6': ('screen_capture_example.py', 'Screen Capture - Live desktop region'),
        '7': ('ipcamera_example.py', 'IP Camera - RTSP/HTTP stream capture'),
        '8': ('industrial_cameras_example.py', 'Industrial Cameras - Basler, Ximea, etc.'),
        '9': ('realsense_example.py', 'RealSense Camera - RGB + depth processing'),
        '10': ('multiple_cameras_example.py', 'Multiple Cameras - Concurrent capture'),
    }
    
    print("FrameSource Examples")
    print("=" * 50)
    
    for key, (filename, description) in examples.items():
        print(f"{key:2}. {description}")
    
    print("\nSelect an example to run (1-10), or 'q' to quit:")
    
    while True:
        choice = input("> ").strip()
        
        if choice.lower() == 'q':
            print("Goodbye!")
            sys.exit(0)
        
        if choice in examples:
            filename, description = examples[choice]
            print(f"\nRunning: {description}")
            print(f"File: {filename}")
            print("-" * 50)
            
            try:
                # Run the selected example
                result = subprocess.run([sys.executable, filename], cwd=os.path.dirname(__file__))
                if result.returncode != 0:
                    print(f"\nExample exited with code {result.returncode}")
                else:
                    print(f"\nExample completed successfully")
            except KeyboardInterrupt:
                print("\nExample interrupted by user")
            except Exception as e:
                print(f"\nError running example: {e}")
            
            print("\nSelect another example or 'q' to quit:")
        else:
            print("Invalid choice. Please select 1-10 or 'q' to quit.")


if __name__ == "__main__":
    main()
