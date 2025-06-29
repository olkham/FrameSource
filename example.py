from frame_source import FrameSourceFactory
import cv2
from typing import Any, List


def test_camera(name, **kwargs):
    # Example 1: Webcam capture
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    print("Testing Webcam Capture:")
    camera = FrameSourceFactory.create(name, **kwargs)
    
    camera.connect()

    threaded = kwargs.get('threaded', False)
    if threaded:
        camera.start()

    if camera.is_connected:

        exposure_range = camera.get_exposure_range()
        if exposure_range is not None:
            min_exp, max_exp = exposure_range
        else:
            min_exp, max_exp = None, None
            
        gain_range = camera.get_gain_range()
        if gain_range is not None:
            min_gain, max_gain = gain_range
        else:
            min_gain, max_gain = None, None

        # Lock exposure time but allow gain to vary for auto exposure
        try:
            # Enable auto gain only while keeping exposure fixed
            camera.enable_auto_exposure(True)  # Enable auto exposure/gain
            
            print("Auto exposure/gain configured: exposure locked, gain variable")
        except Exception as e:
            print(f"Error configuring Ximea auto exposure/gain: {e}")

        # camera.enable_auto_exposure(True)
        print(f"Exposure: {camera.get_exposure()}")
        print(f"Gain: {camera.get_gain()}")
        print(f"Frame size: {camera.get_frame_size()}")
        
        # Read a few frames
        while camera.is_connected:
            ret, frame = camera.read()
            if ret:
                if frame is not None:
                    cv2.imshow("camera", frame)
                # Add key controls for exposure and gain adjustment
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('=') or key == ord('+'):  # Increase exposure
                    current_exposure = camera.get_exposure()
                    if current_exposure is not None and min_exp is not None and max_exp is not None:
                        new_exposure = min(current_exposure + 1000, max_exp)  # Increase by 1ms
                        camera.set_exposure(new_exposure)
                        print(f"Exposure increased to: {new_exposure} (range: {min_exp}-{max_exp})")
                elif key == ord('-'):  # Decrease exposure
                    current_exposure = camera.get_exposure()
                    if current_exposure is not None and min_exp is not None and max_exp is not None:
                        new_exposure = max(current_exposure - 1000, min_exp)  # Decrease by 1ms
                        camera.set_exposure(new_exposure)
                        print(f"Exposure decreased to: {new_exposure} (range: {min_exp}-{max_exp})")
                elif key == ord(']'):  # Increase gain
                    current_gain = camera.get_gain()
                    if current_gain is not None and min_gain is not None and max_gain is not None:
                        new_gain = min(current_gain + 1, max_gain)
                        camera.set_gain(new_gain)
                        print(f"Gain increased to: {new_gain} (range: {min_gain}-{max_gain})")
                elif key == ord('['):  # Decrease gain
                    current_gain = camera.get_gain()
                    if current_gain is not None and min_gain is not None and max_gain is not None:
                        new_gain = max(current_gain - 1, min_gain)
                        camera.set_gain(new_gain)
                        print(f"Gain decreased to: {new_gain} (range: {min_gain}-{max_gain})")
                elif key == ord('a'):  # Toggle auto exposure
                    print("Toggling auto exposure...")
                    camera.enable_auto_exposure(True)
                elif key == ord('m'):  # Manual exposure mode
                    print("Switching to manual exposure...")
                    camera.enable_auto_exposure(False)
                elif key == ord('h'):  # Show help
                    print("\nKey controls:")
                    print("  q - Quit")
                    print("  + or = - Increase exposure")
                    print("  - - Decrease exposure")
                    print("  ] - Increase gain")
                    print("  [ - Decrease gain")
                    print("  a - Enable auto exposure")
                    print("  m - Manual exposure mode")
                    print("  h - Show this help")
            else:
                print(f"Failed to read frame")

    camera.disconnect()


def test_multiple_cameras(cameras:List[Any], threaded:bool = True):
    """Test connecting to multiple different cameras types and viewing them live concurrently."""
    

    capture_instances = []
    grid_cols = 3
    grid_rows = 2
    win_w, win_h = 640, 480
    for idx, cam_cfg in enumerate(cameras):
        name = cam_cfg.pop('capture_type', None)
        if not name:
            print(f"Camera config missing 'capture_type': {cam_cfg}")
            continue
        cv2.namedWindow(f"{name}", cv2.WINDOW_NORMAL)
        # Set window size and position for grid
        cv2.resizeWindow(f"{name}", win_w, win_h)
        col = idx % grid_cols
        row = idx // grid_cols
        x = col * win_w
        y = row * win_h
        cv2.moveWindow(f"{name}", x, y+(25* row))  # Add some vertical spacing
        print(f"Testing {name} Capture:")
        camera = FrameSourceFactory.create(name, **cam_cfg)
        if camera.connect():
            camera.enable_auto_exposure(True)  # Enable auto exposure by default
            if threaded:
                camera.start()  # Always use threaded capture for this test
            capture_instances.append((name, camera))
            print(f"Connected to {name} camera")
        else:
            print(f"Failed to connect to {name} camera")

    try:
        while True:
            for name, camera in capture_instances:
                if camera.is_connected:
                    ret, frame = camera.read()
                    if ret:
                        cv2.imshow(f"{name}", frame)
                    else:
                        print(f"Failed to read frame from {name}")
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        for name, camera in capture_instances:
            if camera.is_connected:
                camera.stop()
            camera.disconnect()
            print(f"Disconnected from {name}")


# Example usage and testing
if __name__ == "__main__":
    # test_camera('basler')
    # test_camera('ximea')
    # test_camera('webcam', source=0)
    # test_camera('video_file', source="media/geti_demo.mp4", loop=True)
    # test_camera('ipcam', source="rtsp://192.168.1.153:554/h264Preview_01_sub", username="admin", password="password")
    # test_camera('folder', source="media/image_seq", sort_by='date', fps=30, real_time=True, loop=True)
    # test_camera('screen', x=100, y=100, w=800, h=600, fps=30, threaded=True)

    cameras = [
        {'capture_type': 'basler', 'threaded': True},
        {'capture_type': 'ximea', 'threaded': True},
        {'capture_type': 'webcam', 'threaded': True},
        # {'capture_type': 'ipcam', 'source': "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg", 'threaded': True},
        # {'capture_type': 'video_file', 'source': "media/geti_demo.mp4", 'loop': True, 'threaded': True},
        # {'capture_type': 'folder', 'source': "media/image_seq", 'sort_by': 'date', 'fps': 30, 'real_time': True, 'loop': True, 'threaded': False},
        # {'capture_type': 'screen', 'x': 100, 'y': 100, 'w': 800, 'h': 600, 'fps': 30, 'threaded': True}
    ]

    test_multiple_cameras(cameras, threaded=True)