#!/usr/bin/env python3
"""
RealSense Camera Example

Demonstrates RealSense camera capture with depth processing capabilities.
Shows RGB, depth, and aligned views.
"""

import cv2
from frame_source.realsense_capture import RealsenseCapture
from frame_processors.realsense_depth_processor import RealsenseDepthProcessor, RealsenseProcessingOutput


def main():
    """Test RealSense camera with depth processing."""
    cv2.namedWindow("RealSense Camera", cv2.WINDOW_NORMAL)
    print("Testing RealSense Camera:")
    
    # Create RealSense capture
    camera = RealsenseCapture(width=640, height=480)
    
    # Create depth processor for side-by-side view
    processor = RealsenseDepthProcessor(output_format=RealsenseProcessingOutput.ALIGNED_SIDE_BY_SIDE)
    camera.attach_processor(processor)
    
    if not camera.connect():
        print("Failed to connect to RealSense camera")
        print("Make sure a RealSense camera is connected and drivers are installed")
        return
    
    if camera.is_connected:
        print(f"Frame size: {camera.get_frame_size()}")
        print(f"FPS: {camera.get_fps()}")
        
        def print_help():
            print("\nRealSense Camera Controls:")
            print("  ESC or q - Quit")
            print("  h - Show this help")
            print("  1 - RGB only")
            print("  2 - Depth only (grayscale)")
            print("  3 - Depth colorized")
            print("  4 - Aligned side-by-side (RGB + Depth)")
            print("  5 - RGBD combined")
        
        print_help()
        
        while camera.is_connected:
            ret, frame = camera.read()
            if ret and frame is not None:
                cv2.imshow("RealSense Camera", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):  # ESC or q to quit
                break
            elif key == ord('h'):
                print_help()
            elif key == ord('1'):  # RGB only
                processor.output_format = RealsenseProcessingOutput.RGB
                print("Output: RGB only")
            elif key == ord('2'):  # Depth only (grayscale)
                processor.output_format = RealsenseProcessingOutput.ALIGNED_DEPTH
                print("Output: Depth only (grayscale)")
            elif key == ord('3'):  # Depth colorized
                processor.output_format = RealsenseProcessingOutput.ALIGNED_DEPTH_COLORIZED
                print("Output: Depth colorized")
            elif key == ord('4'):  # Side-by-side
                processor.output_format = RealsenseProcessingOutput.ALIGNED_SIDE_BY_SIDE
                print("Output: Aligned side-by-side")
            elif key == ord('5'):  # RGBD
                processor.output_format = RealsenseProcessingOutput.RGBD
                print("Output: RGBD combined")
    
    camera.disconnect()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
