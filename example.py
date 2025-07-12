import cv2
from typing import Any, List
from frame_source import FrameSourceFactory
from frame_processors.equirectangular360_processor import Equirectangular2PinholeProcessor


def test_audio_spectrogram(source=None, **kwargs):
    """Test audio spectrogram capture from microphone or audio file."""
    cv2.namedWindow("Audio Spectrogram", cv2.WINDOW_NORMAL)
    print("Testing Audio Spectrogram Capture:")
    
    # Default audio parameters for good visualization
    audio_params = {
        'n_mels': 128,
        'n_fft': 2048,
        'window_duration': 3.0,
        'freq_range': (20, 8000),
        'frame_rate': 30,
        'colormap': cv2.COLORMAP_VIRIDIS,
        **kwargs
    }
    
    frame_source = FrameSourceFactory.create('audio_spectrogram', source=source, **audio_params)
    
    if not frame_source.connect():
        print("Failed to connect to audio source")
        return
    
    threaded = kwargs.get('threaded', True)  # Default to threaded mode
    print(f"Running in {'threaded' if threaded else 'blocking'} mode")
    
    if threaded:
        frame_source.start()
        print("Started background spectrogram capture thread")

    if frame_source.is_connected:
        print(f"Audio spectrogram params:")
        print(f"  Frame size: {frame_source.get_frame_size()}")
        print(f"  FPS: {frame_source.get_fps()}")
        # Audio-specific parameters (type check since not all cameras have these)
        if hasattr(frame_source, 'get_n_mels'):
            print(f"  N mels: {frame_source.get_n_mels()}") # type: ignore
        if hasattr(frame_source, 'get_window_duration'):
            print(f"  Window duration: {frame_source.get_window_duration()}s") # type: ignore
        if hasattr(frame_source, 'get_freq_range'):
            print(f"  Frequency range: {frame_source.get_freq_range()}") # type: ignore
        if hasattr(frame_source, 'get_sample_rate'):
            print(f"  Sample rate: {frame_source.get_sample_rate()}Hz") # type: ignore
        if hasattr(frame_source, 'get_nyquist_frequency'):
            print(f"  Nyquist frequency (max): {frame_source.get_nyquist_frequency()}Hz") # type: ignore
        if hasattr(frame_source, 'get_fft_size'):
            print(f"  FFT size: {frame_source.get_fft_size()}") # type: ignore
        if hasattr(frame_source, 'get_contrast_method'):
            print(f"  Contrast method: {frame_source.get_contrast_method()}") # type: ignore
        if hasattr(frame_source, 'get_gamma_correction'):
            print(f"  Gamma correction: {frame_source.get_gamma_correction():.2f}") # type: ignore
        if hasattr(frame_source, 'get_noise_floor'):
            print(f"  Noise floor: {frame_source.get_noise_floor()} dB") # type: ignore
        if hasattr(frame_source, 'get_percentile_range'):
            print(f"  Percentile range: {frame_source.get_percentile_range()}%") # type: ignore
        
        def print_help():
            print("\nKey controls:")
            print("  ESC - Quit")
            print("  h - Show this help")
            print("  0 - Grayscale (default)")
            print("  1 - Viridis colormap")
            print("  2 - Plasma colormap")
            print("  3 - Inferno colormap") 
            print("  4 - Hot colormap")
            print("  5 - Jet colormap")
            print("  +/- - Adjust mel bands (requires restart)")
            print("  c - Cycle contrast methods (fixed/adaptive/percentile)")
            print("  g/G - Decrease/increase gamma correction")
            print("  n/N - Decrease/increase noise floor")
            print("  p/P - Adjust percentile range")
        
        print_help()
        
        while frame_source.is_connected:
            ret, frame = frame_source.read()
            if ret and frame is not None:
                cv2.imshow("Audio Spectrogram", frame)
                
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key to quit
                break
            elif key == ord('h'):  # Show help
                print_help()
            elif key == ord('0') and hasattr(frame_source, 'set_colormap'):
                frame_source.set_colormap(None) # type: ignore
                print("Colormap: Grayscale")
            elif key == ord('1') and hasattr(frame_source, 'set_colormap'):
                frame_source.set_colormap(cv2.COLORMAP_VIRIDIS) # type: ignore
                print("Colormap: Viridis")
            elif key == ord('2') and hasattr(frame_source, 'set_colormap'):
                frame_source.set_colormap(cv2.COLORMAP_PLASMA) # type: ignore
                print("Colormap: Plasma")
            elif key == ord('3') and hasattr(frame_source, 'set_colormap'):
                frame_source.set_colormap(cv2.COLORMAP_INFERNO) # type: ignore
                print("Colormap: Inferno")
            elif key == ord('4') and hasattr(frame_source, 'set_colormap'):
                frame_source.set_colormap(cv2.COLORMAP_HOT) # type: ignore
                print("Colormap: Hot")
            elif key == ord('5') and hasattr(frame_source, 'set_colormap'):
                frame_source.set_colormap(cv2.COLORMAP_JET) # type: ignore
                print("Colormap: Jet")
            elif key == ord('+') or key == ord('='):
                if hasattr(frame_source, 'get_n_mels') and hasattr(frame_source, 'set_n_mels'):
                    current_mels = frame_source.get_n_mels() # type: ignore
                    frame_source.set_n_mels(min(current_mels + 16, 256)) # type: ignore
                    print(f"Mel bands: {frame_source.get_n_mels()} (restart to apply)") # type: ignore
            elif key == ord('-'):
                if hasattr(frame_source, 'get_n_mels') and hasattr(frame_source, 'set_n_mels'):
                    current_mels = frame_source.get_n_mels() # type: ignore
                    frame_source.set_n_mels(max(current_mels - 16, 32)) # type: ignore
                    print(f"Mel bands: {frame_source.get_n_mels()} (restart to apply)") # type: ignore
            elif key == ord('c'):  # Cycle contrast methods
                if hasattr(frame_source, 'get_contrast_method') and hasattr(frame_source, 'set_contrast_method'):
                    current_method = frame_source.get_contrast_method() # type: ignore
                    methods = ['fixed', 'adaptive', 'percentile']
                    current_index = methods.index(current_method)
                    next_method = methods[(current_index + 1) % len(methods)]
                    frame_source.set_contrast_method(next_method) # type: ignore
                    print(f"Contrast method: {next_method}")
            elif key == ord('g'):  # Decrease gamma
                if hasattr(frame_source, 'get_gamma_correction') and hasattr(frame_source, 'set_gamma_correction'):
                    current_gamma = frame_source.get_gamma_correction() # type: ignore
                    new_gamma = max(current_gamma - 0.1, 0.1)
                    frame_source.set_gamma_correction(new_gamma) # type: ignore
                    print(f"Gamma correction: {new_gamma:.2f} ({'more contrast' if new_gamma < 1.0 else 'less contrast'})")
            elif key == ord('G'):  # Increase gamma
                if hasattr(frame_source, 'get_gamma_correction') and hasattr(frame_source, 'set_gamma_correction'):
                    current_gamma = frame_source.get_gamma_correction() # type: ignore
                    new_gamma = min(current_gamma + 0.1, 3.0)
                    frame_source.set_gamma_correction(new_gamma) # type: ignore
                    print(f"Gamma correction: {new_gamma:.2f} ({'more contrast' if new_gamma < 1.0 else 'less contrast'})")
            elif key == ord('n'):  # Decrease noise floor (less noise suppression)
                if hasattr(frame_source, 'get_noise_floor') and hasattr(frame_source, 'set_noise_floor'):
                    current_floor = frame_source.get_noise_floor() # type: ignore
                    new_floor = max(current_floor - 5, -100)
                    frame_source.set_noise_floor(new_floor) # type: ignore
                    print(f"Noise floor: {new_floor} dB")
            elif key == ord('N'):  # Increase noise floor (more noise suppression)
                if hasattr(frame_source, 'get_noise_floor') and hasattr(frame_source, 'set_noise_floor'):
                    current_floor = frame_source.get_noise_floor() # type: ignore
                    new_floor = min(current_floor + 5, -10)
                    frame_source.set_noise_floor(new_floor) # type: ignore
                    print(f"Noise floor: {new_floor} dB")
            elif key == ord('p'):  # Decrease percentile range (more aggressive)
                if hasattr(frame_source, 'get_percentile_range') and hasattr(frame_source, 'set_percentile_range'):
                    low, high = frame_source.get_percentile_range() # type: ignore
                    new_low = min(low + 2, 20)
                    new_high = max(high - 2, 80)
                    if new_low < new_high:
                        frame_source.set_percentile_range(new_low, new_high) # type: ignore
                        print(f"Percentile range: {new_low}-{new_high}% (more aggressive)")
            elif key == ord('P'):  # Increase percentile range (less aggressive)
                if hasattr(frame_source, 'get_percentile_range') and hasattr(frame_source, 'set_percentile_range'):
                    low, high = frame_source.get_percentile_range() # type: ignore
                    new_low = max(low - 2, 0)
                    new_high = min(high + 2, 100)
                    frame_source.set_percentile_range(new_low, new_high) # type: ignore
                    print(f"Percentile range: {new_low}-{new_high}% (less aggressive)")

    if threaded:
        frame_source.stop()
        print("Stopped background spectrogram capture thread")
    
    frame_source.disconnect()
    cv2.destroyWindow("Audio Spectrogram")


def test_360_camera(name, **kwargs):
    """Test a 360 camera with equirectangular to pinhole projection."""
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    print("Testing 360 Camera Capture:")
    camera = FrameSourceFactory.create(name, **kwargs)
    camera.connect()
    
    # Set camera resolution and fps - insta360 x5 webcam mode settings
    camera.set_frame_size(2880, 1440)
    camera.set_fps(30)
    
    # Add processor if specified
    if 'processor' in kwargs:
        processor_config = kwargs.pop('processor')
        processor_type = processor_config.pop('type')
        if processor_type == 'equirectangular':
            processor = Equirectangular2PinholeProcessor(**processor_config)
            camera.attach_processor(processor)

    threaded = kwargs.get('threaded', False)
    if threaded:
        camera.start()

    if camera.is_connected:
        print(f"Frame size: {camera.get_frame_size()}")
        print(f"FPS: {camera.get_fps()}")
        
        # Read a few frames
        while camera.is_connected:
            ret, frame = camera.read()
            if ret:
                if frame is not None:
                    cv2.imshow("camera", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key to quit
                    break
                elif key == ord('h'):  # Show help
                    print("\nKey controls:")
                    print("  ESC - Quit")
                    print("  h - Show this help")
                    # Add processor controls if processor is attached
                    if hasattr(camera, '_processors') and camera._processors:
                        print("\nProcessor controls:")
                        print("  w/s - Adjust pitch (up/down)")
                        print("  a/d - Adjust yaw (left/right)")  
                        print("  q/e - Adjust roll (left/right)")
                        print("  r - Reset processor angles")
                elif key == ord('w'):  # Pitch up
                    processors = getattr(camera, '_processors', [])
                    for processor in processors:
                        if hasattr(processor, 'get_parameter'):
                            current_pitch = processor.get_parameter('pitch') or 0
                            processor.set_parameter('pitch', current_pitch + 5.0)
                            print(f"Pitch: {processor.get_parameter('pitch'):.1f}°")
                elif key == ord('s'):  # Pitch down
                    processors = getattr(camera, '_processors', [])
                    for processor in processors:
                        if hasattr(processor, 'get_parameter'):
                            current_pitch = processor.get_parameter('pitch') or 0
                            processor.set_parameter('pitch', current_pitch - 5.0)
                            print(f"Pitch: {processor.get_parameter('pitch'):.1f}°")
                elif key == ord('a'):  # Yaw left
                    processors = getattr(camera, '_processors', [])
                    for processor in processors:
                        if hasattr(processor, 'get_parameter'):
                            current_yaw = processor.get_parameter('yaw') or 0
                            processor.set_parameter('yaw', current_yaw - 5.0)
                            print(f"Yaw: {processor.get_parameter('yaw'):.1f}°")
                elif key == ord('d'):  # Yaw right
                    processors = getattr(camera, '_processors', [])
                    for processor in processors:
                        if hasattr(processor, 'get_parameter'):
                            current_yaw = processor.get_parameter('yaw') or 0
                            processor.set_parameter('yaw', current_yaw + 5.0)
                            print(f"Yaw: {processor.get_parameter('yaw'):.1f}°")
                elif key == ord('q'):  # Roll left
                    processors = getattr(camera, '_processors', [])
                    for processor in processors:
                        if hasattr(processor, 'get_parameter'):
                            current_roll = processor.get_parameter('roll') or 0
                            processor.set_parameter('roll', current_roll - 5.0)
                            print(f"Roll: {processor.get_parameter('roll'):.1f}°")
                elif key == ord('e'):  # Roll right
                    processors = getattr(camera, '_processors', [])
                    for processor in processors:
                        if hasattr(processor, 'get_parameter'):
                            current_roll = processor.get_parameter('roll') or 0
                            processor.set_parameter('roll', current_roll + 5.0)
                            print(f"Roll: {processor.get_parameter('roll'):.1f}°")
                elif key == ord('r'):  # Reset processor angles
                    processors = getattr(camera, '_processors', [])
                    for processor in processors:
                        if hasattr(processor, 'set_parameter'):
                            processor.set_parameter('pitch', 0.0)
                            processor.set_parameter('yaw', 0.0)
                            processor.set_parameter('roll', 0.0)
                            print("Processor angles reset to 0°")
            else:
                print(f"Failed to read frame")

    camera.disconnect() 


def test_camera(name, **kwargs):
    # Example 1: Webcam capture
    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    print("Testing Webcam Capture:")
    camera = FrameSourceFactory.create(name, **kwargs)
    camera.connect()

    threaded = kwargs.get('threaded', False)
    if threaded:
        camera.start()

    width = kwargs.get('width', 1920)
    height = kwargs.get('height', 1080)
    fps = kwargs.get('fps', 30)
    camera.set_frame_size(width, height)    
    camera.set_fps(fps)
    
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
    
    test_audio_spectrogram(source=None, threaded=True, n_mels=256, window_duration=5.0, freq_range=(20, 20000),
                           sample_rate=44100, db_range=(-60, 0), contrast_method='adaptive', 
                           gamma_correction=0.7, noise_floor=-65, percentile_range=(10, 90))
    
    # test_camera('basler')
    # test_camera('ximea')
    # test_camera('webcam', source=0, threaded=True, width=1920, height=1080, fps=30)   # standard 1080p webcam
    # test_camera('webcam', source=0, threaded=True, width=2880, height=1440, fps=30)   # insta360 x5 webcam mode settings
    # test_camera('video_file', source="media/geti_demo.mp4", loop=True)
    # test_camera('ipcam', source="rtsp://192.168.1.153:554/h264Preview_01_sub", username="admin", password="password")
    # test_camera('folder', source="media/image_seq", sort_by='date', fps=30, real_time=True, loop=True)
    # test_camera('screen', x=100, y=100, w=800, h=600, fps=30, threaded=True)

    # test_360_camera('webcam', source=0, threaded=True, processor={'type': 'equirectangular', 'output_width': 1920, 'output_height': 1080, 'fov': 90})

    # cameras = [
        # {'capture_type': 'basler', 'threaded': True},
        # {'capture_type': 'ximea', 'threaded': True},
        # {'capture_type': 'webcam', 'threaded': True},
        # {'capture_type': 'ipcam', 'source': "http://pendelcam.kip.uni-heidelberg.de/mjpg/video.mjpg", 'threaded': True},
        # {'capture_type': 'video_file', 'source': "media/geti_demo.mp4", 'loop': True, 'threaded': True},
        # {'capture_type': 'folder', 'source': "media/image_seq", 'sort_by': 'date', 'fps': 30, 'real_time': True, 'loop': True, 'threaded': False},
        # {'capture_type': 'screen', 'x': 100, 'y': 100, 'w': 800, 'h': 600, 'fps': 30, 'threaded': True}
    # ]

    # test_multiple_cameras(cameras, threaded=True)