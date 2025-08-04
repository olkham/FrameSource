# FrameSource Examples

This directory contains organized examples for all FrameSource capture types. Each example demonstrates a specific frame source with interactive controls and best practices.

## Available Examples

### 1. **webcam_example.py** - Webcam Capture
- Basic webcam capture with manual controls
- Exposure and gain adjustment
- Auto exposure toggle
- Supports threading for smooth capture

### 2. **audio_spectrogram_example.py** - Audio Spectrogram
- Convert microphone or audio file to visual spectrogram
- Real-time contrast adjustment (adaptive, percentile, fixed)
- Multiple colormap options
- Gamma correction and noise floor controls
- Interactive parameter tuning

### 3. **camera_360_example.py** - 360° Camera Processing
- Equirectangular to pinhole projection
- Interactive view controls (pitch, yaw, roll)
- FOV adjustment
- Perfect for 360° cameras like Insta360

### 4. **video_file_example.py** - Video File Playback
- Video file playback with looping
- Pause/resume controls
- Restart functionality
- Real-time frame rate display

### 5. **image_folder_example.py** - Image Sequence
- Play image sequences from folders
- Configurable FPS and sorting (name/date)
- Real-time playback with controls
- Loop and restart functionality

### 6. **screen_capture_example.py** - Screen Capture
- Live desktop region capture
- Configurable capture area
- FPS adjustment
- Perfect for recording applications

### 7. **ipcamera_example.py** - IP Camera Streaming
- RTSP and HTTP stream support
- Automatic reconnection handling
- Authentication support
- Network error recovery

### 8. **industrial_cameras_example.py** - Industrial Cameras
- Basler and Ximea camera support
- Professional exposure/gain controls
- Auto exposure modes
- High-performance imaging

### 9. **realsense_example.py** - RealSense Depth Camera
- RGB and depth capture
- Multiple output formats (RGB, depth, side-by-side, overlay)
- Real-time depth processing
- Intel RealSense integration

### 10. **multiple_cameras_example.py** - Multi-Camera Setup
- Concurrent capture from multiple sources
- Grid window layout
- Mixed camera types support
- Centralized control and monitoring

## Running Examples

### Option 1: Interactive Menu
```bash
cd examples
python run_examples.py
```

### Option 2: Direct Execution
```bash
cd examples
python webcam_example.py
python audio_spectrogram_example.py
# ... etc
```

### Option 3: From Parent Directory
```bash
python -m examples.webcam_example
python -m examples.audio_spectrogram_example
# ... etc
```

## Requirements

Different examples may require additional dependencies:

- **Audio examples**: `pip install .[audio]` (librosa, soundfile, pyaudio)
- **RealSense examples**: `pip install .[realsense]` (pyrealsense2)
- **All features**: `pip install .[full]`

## Common Controls

Most examples share these common keyboard controls:
- **ESC** or **q** - Quit the example
- **h** - Show help/controls
- **Space** - Pause/resume (where applicable)

Specific controls are documented in each example's help system (press 'h' while running).

## Customization

Each example is designed to be easily customizable:
1. Modify camera parameters at the top of each file
2. Adjust window sizes and positions
3. Change default settings for your hardware
4. Add custom processing or overlays

## Troubleshooting

- **Camera not found**: Check device connections and drivers
- **Permission errors**: Ensure camera access permissions
- **Import errors**: Install required dependencies with pip
- **Performance issues**: Try reducing resolution or FPS
- **Network cameras**: Verify URLs and network connectivity

## Adding New Examples

To add a new example:
1. Create a new Python file following the naming pattern
2. Use the existing examples as templates
3. Add entry to `run_examples.py`
4. Update this README with description
