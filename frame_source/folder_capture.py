import os
import cv2
import numpy as np
import time
from typing import Optional, Tuple, List
from .video_capture_base import VideoCaptureBase
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FolderCapture(VideoCaptureBase):
    """
    Capture class for reading images from a folder as a video stream.
    Images can be sorted by creation time or by name.
    """
    def __init__(self, source: str, sort_by: str = 'name', width: Optional[int] = None, height: Optional[int] = None, fps: float = 30.0, real_time: bool = True, loop: bool = False, **kwargs):
        super().__init__(source, **kwargs)
        self.sort_by = sort_by
        self.width = width
        self.height = height
        self.fps = fps
        self.real_time = real_time
        self.loop = loop
        self.image_files: List[str] = []
        self.index = 0
        self.time_of_last_frame = 0.0
        self._capture_thread = None
        self._stop_event = None
        self._latest_frame = None
        self._latest_success = False

    def start(self):
        """
        Start background frame capture in a separate thread.
        Continuously updates self._latest_frame and self._latest_success.
        """
        import threading
        import time
        if hasattr(self, '_capture_thread') and self._capture_thread and self._capture_thread.is_alive():
            return  # Already running
        self._stop_event = threading.Event()
        self._latest_frame = None
        self._latest_success = False
        def _capture_loop():
            while not self._stop_event.is_set(): # type: ignore
                success, frame = self._read_frame_for_thread()
                self._latest_success = success
                self._latest_frame = frame
                time.sleep(0.01)  # 10ms delay to avoid busy loop
        self._capture_thread = threading.Thread(target=_capture_loop, daemon=True)
        self._capture_thread.start()

    def stop(self):
        """
        Stop background frame capture thread.
        """
        if hasattr(self, '_stop_event') and self._stop_event:
            self._stop_event.set()
        if hasattr(self, '_capture_thread') and self._capture_thread:
            self._capture_thread.join(timeout=1)
        self._capture_thread = None
        self._stop_event = None

    def _read_frame_for_thread(self):
        """
        Internal: Calls the read() method for background thread use.
        """
        return self._read_direct()

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the most recent frame captured by the background thread.
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        return getattr(self, '_latest_success', False), getattr(self, '_latest_frame', None)

    def connect(self) -> bool:
        if not os.path.isdir(self.source):
            logger.error(f"Folder not found: {self.source}")
            return False
        # List image files
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        files = [os.path.join(self.source, f) for f in os.listdir(self.source) if f.lower().endswith(valid_exts)]
        if self.sort_by == 'date':
            files.sort(key=lambda x: os.path.getctime(x))
        else:
            files.sort()  # Default: sort by name
        if not files:
            logger.error(f"No image files found in folder: {self.source}")
            return False
        self.image_files = files
        self.index = 0
        self.is_connected = True
        self.time_of_last_frame = time.time()
        logger.info(f"Connected to folder with {len(files)} images.")
        return True

    def disconnect(self) -> bool:
        self.is_connected = False
        self.image_files = []
        self.index = 0
        logger.info("Disconnected from folder capture.")
        return True

    def _read_implementation(self) -> Tuple[bool, Optional[np.ndarray]]:
        # If background thread is running, return latest frame
        if hasattr(self, '_capture_thread') and self._capture_thread is not None and self._capture_thread.is_alive():
            return self.get_latest_frame()
        else:
            return self._read_direct()

    def _read_direct(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_connected or not self.image_files:
            return False, None
        # Real-time playback control
        if self.real_time and self.fps > 0:
            frame_duration = 1.0 / self.fps
            now = time.time()
            elapsed = now - self.time_of_last_frame
            if elapsed < frame_duration:
                time.sleep(frame_duration - elapsed)
            self.time_of_last_frame = time.time()
        # Read image
        if self.index >= len(self.image_files):
            if self.loop:
                self.index = 0
            else:
                return False, None
        img_path = self.image_files[self.index]
        img = cv2.imread(img_path)
        if img is None:
            logger.warning(f"Failed to read image: {img_path}")
            self.index += 1
            return False, None
        # Resize if needed
        if self.width is not None and self.height is not None:
            img = cv2.resize(img, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.index += 1
        return True, img

    def set_exposure(self, value: float) -> bool:
        logger.warning("Exposure control not applicable for folder capture.")
        return False

    def get_exposure(self) -> Optional[float]:
        return None

    def set_gain(self, value: float) -> bool:
        logger.warning("Gain control not applicable for folder capture.")
        return False

    def get_gain(self) -> Optional[float]:
        return None

    def enable_auto_exposure(self, enable: bool = True) -> bool:
        logger.warning("Auto exposure control not applicable for folder capture.")
        return False

    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        if not self.image_files:
            return None
        img = cv2.imread(self.image_files[0])
        if img is None:
            return None
        h, w = img.shape[:2]
        if self.width is not None and self.height is not None:
            return (self.width, self.height)
        return (w, h)

    def set_frame_size(self, width: int, height: int) -> bool:
        self.width = width
        self.height = height
        return True

    def get_fps(self) -> Optional[float]:
        return self.fps

    def set_fps(self, fps: float) -> bool:
        self.fps = fps
        return True

if __name__ == "__main__":
    import sys
    folder = "C:/Users/optane/Desktop/bird-calls-dataset/images/default"
    cap = FolderCapture(folder, sort_by='date', fps=10, real_time=True, loop=True)
    if cap.connect():
        # cap.start()
        cv2.namedWindow("FolderCapture", cv2.WINDOW_NORMAL)
        while cap.is_connected:
            ret, frame = cap.read()
            if ret:
                if frame is not None:
                    cv2.imshow("FolderCapture", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            else:
                break
        # cap.stop()
        cap.disconnect()
        cv2.destroyAllWindows()
    else:
        print("Failed to connect to folder.")
