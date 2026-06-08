#!/usr/bin/env python3
"""
Test OpenCV-compatible API methods: isOpened() and release()
This demonstrates that FrameSource capture objects can now be used
similarly to cv2.VideoCapture objects.
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_opencv_api():
    """Test that the OpenCV-compatible API works correctly."""
    
    print("Testing OpenCV-compatible API methods")
    print("=" * 80)
    
    # Import and create a capture instance
    try:
        from framesource import FrameSourceFactory
        
        # Try to create a webcam capture
        print("\n1️⃣ Creating WebcamCapture instance...")
        cap = FrameSourceFactory.create('webcam', source=0)
        print(f"   ✅ Created: {cap}")
        
        # Test isOpened() before connecting
        print("\n2️⃣ Testing isOpened() before connect()...")
        is_open = cap.isOpened()
        print(f"   Status: {is_open}")
        assert not is_open, "Should be False before connecting"
        print("   ✅ Correct: Returns False when not connected")
        
        # Connect
        print("\n3️⃣ Connecting to device...")
        try:
            success = cap.connect()
            if success:
                print("   ✅ Connected successfully")
            else:
                print("   ⚠️  Connection failed (device may not be available)")
        except Exception as e:
            print(f"   ⚠️  Connection error: {e}")
            success = False
        
        # Test isOpened() after connecting
        print("\n4️⃣ Testing isOpened() after connect()...")
        is_open = cap.isOpened()
        print(f"   Status: {is_open}")
        if success:
            assert is_open, "Should be True after successful connection"
            print("   ✅ Correct: Returns True when connected")
        else:
            print("   ⚠️  Not connected, skipping assertion")
        
        # Test release()
        print("\n5️⃣ Testing release()...")
        cap.release()
        print("   ✅ release() called successfully")
        
        # Test isOpened() after release
        print("\n6️⃣ Testing isOpened() after release()...")
        is_open = cap.isOpened()
        print(f"   Status: {is_open}")
        assert not is_open, "Should be False after release"
        print("   ✅ Correct: Returns False after release")
        
        print("\n" + "=" * 80)
        print("✨ OpenCV-compatible API test complete!")
        print("\nUsage example (OpenCV-style):")
        print("-" * 80)
        print("from framesource import FrameSourceFactory")
        print("")
        print("# Create capture (like cv2.VideoCapture)")
        print("cap = FrameSourceFactory.create('webcam', source=0)")
        print("cap.connect()")
        print("")
        print("# Check if opened (like cv2.VideoCapture.isOpened())")
        print("if cap.isOpened():")
        print("    ret, frame = cap.read()")
        print("    if ret:")
        print("        # Process frame...")
        print("        pass")
        print("")
        print("# Release resources (like cv2.VideoCapture.release())")
        print("cap.release()")
        print("=" * 80)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure the frame_source package is properly set up")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_opencv_api()
