# FrameSource 📷🖼️

FrameSource is a flexible, extensible Python framework for acquiring frames from a wide variety of sources—webcams, industrial cameras, IP cameras, video files, and even folders of images—using a unified interface. It was created to support my many projects that require switching between different frame providers without changing the downstream frame processing code.

> **Note:** This project was mostly written with the help of GitHub Copilot 🤖, making development fast, fun, and consistent!

## Why FrameSource?

When working on computer vision, robotics, or video analytics projects, you often need to swap between different sources of frames: a webcam for quick tests, a folder of images for batch processing, a video file for reproducibility, or a specialized camera for deployment. FrameSource lets you do this with minimal code changes—just swap the provider!

## Features ✨

- Unified interface for all frame sources (cameras, video files, image folders)
- Easily extensible with new capture types
- Threaded/background capture support for smooth frame acquisition
- Control over exposure, gain, resolution, FPS (where supported)
- Real-time playback and looping for video and image folders
- Simple factory pattern for instantiating sources

## Supported Sources

- 🖥️ **Webcam** (OpenCV)
- 🌐 **IP Camera** (RTSP/HTTP)
- 🎥 **Video File** (MP4, AVI, etc.)
- 🗂️ **Folder of Images** (sorted by name or creation time)
- 🏭 **Industrial Cameras** (e.g., Ximea, Basler)

## Example Usage

### 1. Using the Factory

```python
from video_capture_system import VideoCaptureFactory

# Webcam
cap = VideoCaptureFactory.create('webcam', source=0)
cap.connect()
ret, frame = cap.read()
cap.disconnect()

# Video file
cap = VideoCaptureFactory.create('video_file', source='video.mp4', loop=True)
cap.connect()
while cap.is_connected:
    ret, frame = cap.read()
    if not ret:
        break
cap.disconnect()

# Folder of images
cap = VideoCaptureFactory.create('folder', source='images/', sort_by='date', fps=10, loop=True)
cap.connect()
cap.start()  # For background capture
while cap.is_connected:
    ret, frame = cap.read()
    if not ret:
        break
cap.disconnect()
```

### 2. Direct Use

```python
from folder_capture import FolderCapture
cap = FolderCapture('images/', sort_by='name', width=640, height=480, fps=15, real_time=True, loop=True)
cap.connect()
while cap.is_connected:
    ret, frame = cap.read()
    if not ret:
        break
cap.disconnect()
```

## Extending FrameSource

Want to add a new camera or source? Just subclass `VideoCaptureBase` and register it:

```python
from video_capture_system import VideoCaptureFactory
VideoCaptureFactory.register_capture_type('my_camera', MyCameraCapture)
```

## Credits

- Written by me, with lots of help from GitHub Copilot 🤖
- OpenCV and other camera SDKs for backend support

---

Happy frame grabbing! 🚀
