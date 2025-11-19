"""
Quick test of the window discovery feature.
Run this to verify that window discovery is working correctly.
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from frame_source import ScreenCapture


def test_discovery():
    """Test the discovery feature."""
    print("Testing ScreenCapture.discover()...\n")
    
    sources = ScreenCapture.discover()
    print(sources)
    
    monitors = [s for s in sources if s.get('type') == 'monitor']
    windows = [s for s in sources if s.get('type') == 'window']
    
    print(f"✓ Found {len(monitors)} monitor(s)")
    print(f"✓ Found {len(windows)} window(s)")
    
    if monitors:
        print("\nMonitor details:")
        for m in monitors:
            print(f"  - {m['name']}: {m['width']}x{m['height']} at ({m['left']}, {m['top']})")
    
    if windows:
        print(f"\nFirst 10 windows:")
        for w in windows[:10]:
            print(f"  - {w['title'][:60]:<60} | {w['width']}x{w['height']}")
    else:
        print("\n⚠ No windows discovered. Is pywin32 installed?")
        print("  Install with: pip install pywin32")
    
    print(f"\n✓ Discovery test complete!")
    print(f"  Total sources: {len(sources)}")
    
    return sources


def test_window_capture():
    """Test capturing from a window."""
    print("\n" + "="*60)
    print("Testing window capture...")
    print("="*60)
    
    sources = ScreenCapture.discover()
    windows = [s for s in sources if s.get('type') == 'window']
    
    if not windows:
        print("No windows available to test capture")
        return False
    
    # Try to capture the first window
    window = windows[0]
    print(f"\nTesting capture from: {window['title']}")
    
    try:
        camera = ScreenCapture.from_window(window['hwnd'], fps=30)
        
        if camera.connect():
            print("✓ Successfully connected")
            
            # Try to read one frame
            ret, frame = camera.read()
            if ret and frame is not None:
                print(f"✓ Successfully captured frame: {frame.shape}")
                print(f"  Frame size: {frame.shape[1]}x{frame.shape[0]}")
            else:
                print("✗ Failed to capture frame")
            
            camera.disconnect()
            return True
        else:
            print("✗ Failed to connect")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False


if __name__ == "__main__":
    try:
        # Test discovery
        sources = test_discovery()
        
        # Test window capture if windows are available
        windows = [s for s in sources if s.get('type') == 'window']
        if windows:
            test_window_capture()
        
        print("\n" + "="*60)
        print("All tests complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
