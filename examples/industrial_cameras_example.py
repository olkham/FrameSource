#!/usr/bin/env python3
"""
Industrial Camera Examples

Demonstrates usage of various industrial cameras (Basler, Ximea, etc.)
with exposure and gain controls.
"""

import cv2
from frame_source import FrameSourceFactory


def test_basler_camera():
    """Test Basler camera capture."""
    cv2.namedWindow("Basler Camera", cv2.WINDOW_NORMAL)
    print("Testing Basler Camera:")
    
    camera = FrameSourceFactory.create('basler', threaded=True)
    
    if not camera.connect():
        print("Failed to connect to Basler camera")
        return
    
    camera.start_async()
    run_camera_test(camera, "Basler Camera")


def test_ximea_camera():
    """Test Ximea camera capture."""
    cv2.namedWindow("Ximea Camera", cv2.WINDOW_NORMAL)
    print("Testing Ximea Camera:")
    
    camera = FrameSourceFactory.create('ximea', threaded=True)
    
    if not camera.connect():
        print("Failed to connect to Ximea camera")
        return
    
    camera.start_async()
    run_camera_test(camera, "Ximea Camera")


def run_camera_test(camera, window_name):
    """Common test routine for industrial cameras."""
    if camera.is_connected:
        # Set camera parameters
        camera.set_frame_size(1920, 1080)
        camera.set_fps(30)
        
        # Get parameter ranges
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
            print(f"\n{window_name} Controls:")
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
                cv2.imshow(window_name, frame)
            
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


def main():
    """Test industrial cameras."""
    print("Industrial Camera Examples")
    print("1. Basler Camera")
    print("2. Ximea Camera")
    
    choice = input("Select camera type (1-2, or 'all' for both): ").strip()
    
    if choice == '1':
        test_basler_camera()
    elif choice == '2':
        test_ximea_camera()
    elif choice.lower() == 'all':
        print("Testing all industrial cameras...")
        test_basler_camera()
        test_ximea_camera()
    else:
        print("Invalid choice. Testing Basler camera by default.")
        test_basler_camera()


if __name__ == "__main__":
    main()
