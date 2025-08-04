#!/usr/bin/env python3
"""
Webcam Capture Example

Demonstrates basic webcam capture with manual controls for exposure, gain, and auto exposure.
Supports threaded capture for smooth frame acquisition.
"""

import cv2
from frame_source import FrameSourceFactory


def main():
    """Test webcam capture with interactive controls."""
    cv2.namedWindow("Webcam", cv2.WINDOW_NORMAL)
    print("Testing Webcam Capture:")
    
    # Create webcam capture
    camera = FrameSourceFactory.create('webcam', source=0, threaded=True)
    
    if not camera.connect():
        print("Failed to connect to webcam")
        return
    
    # Configure camera settings
    camera.set_frame_size(1920, 1080)  # Full HD
    camera.set_fps(30)
    camera.start_async()
    
    if camera.is_connected:
        # Get exposure and gain ranges
        exposure_range = camera.get_exposure_range()
        gain_range = camera.get_gain_range()
        
        if exposure_range:
            min_exp, max_exp = exposure_range
        else:
            min_exp, max_exp = None, None
            
        if gain_range:
            min_gain, max_gain = gain_range
        else:
            min_gain, max_gain = None, None
        
        # Enable auto exposure by default
        try:
            camera.enable_auto_exposure(True)
            print("Auto exposure enabled")
        except Exception as e:
            print(f"Auto exposure not supported: {e}")
        
        print(f"Frame size: {camera.get_frame_size()}")
        print(f"FPS: {camera.get_fps()}")
        print(f"Exposure: {camera.get_exposure()}")
        print(f"Gain: {camera.get_gain()}")
        
        def print_help():
            print("\nWebcam Controls:")
            print("  ESC or q - Quit")
            print("  h - Show this help")
            print("  + or = - Increase exposure")
            print("  - - Decrease exposure")
            print("  ] - Increase gain")
            print("  [ - Decrease gain")
            print("  a - Enable auto exposure")
            print("  m - Manual exposure mode")
        
        print_help()
        
        while camera.is_connected:
            ret, frame = camera.read()
            if ret and frame is not None:
                cv2.imshow("Webcam", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q to quit
                break
            elif key == ord('h'):
                print_help()
            elif key == ord('=') or key == ord('+'):  # Increase exposure
                current_exposure = camera.get_exposure()
                if current_exposure is not None and min_exp is not None and max_exp is not None:
                    new_exposure = min(current_exposure + 1000, max_exp)
                    camera.set_exposure(new_exposure)
                    print(f"Exposure: {new_exposure} (range: {min_exp}-{max_exp})")
            elif key == ord('-'):  # Decrease exposure
                current_exposure = camera.get_exposure()
                if current_exposure is not None and min_exp is not None and max_exp is not None:
                    new_exposure = max(current_exposure - 1000, min_exp)
                    camera.set_exposure(new_exposure)
                    print(f"Exposure: {new_exposure} (range: {min_exp}-{max_exp})")
            elif key == ord(']'):  # Increase gain
                current_gain = camera.get_gain()
                if current_gain is not None and min_gain is not None and max_gain is not None:
                    new_gain = min(current_gain + 1, max_gain)
                    camera.set_gain(new_gain)
                    print(f"Gain: {new_gain} (range: {min_gain}-{max_gain})")
            elif key == ord('['):  # Decrease gain
                current_gain = camera.get_gain()
                if current_gain is not None and min_gain is not None and max_gain is not None:
                    new_gain = max(current_gain - 1, min_gain)
                    camera.set_gain(new_gain)
                    print(f"Gain: {new_gain} (range: {min_gain}-{max_gain})")
            elif key == ord('a'):  # Enable auto exposure
                print("Enabling auto exposure...")
                camera.enable_auto_exposure(True)
            elif key == ord('m'):  # Manual exposure
                print("Switching to manual exposure...")
                camera.enable_auto_exposure(False)
    
    camera.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
