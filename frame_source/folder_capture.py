import os
import cv2
import numpy as np
import time
from typing import Optional, Tuple, List
import threading

# Handle both relative imports (when used as module) and absolute imports (when run standalone)
try:
    from .video_capture_base import VideoCaptureBase
except ImportError:
    # For standalone testing, add parent directory to path
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from frame_source.video_capture_base import VideoCaptureBase

import logging

# Try to import watchdog for folder monitoring
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
    WATCHDOG_AVAILABLE = True
except ImportError:
    WATCHDOG_AVAILABLE = False
    print("Warning: watchdog library not available. Install with 'pip install watchdog' for automatic folder monitoring.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FolderWatcherHandler:
    """File system event handler for folder watching."""
    
    def __init__(self, folder_capture_instance):
        self.folder_capture = folder_capture_instance
        self.valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        self.pending_refresh = False
        self.refresh_timer = None
        self.batch_delay = 0.5  # Wait 500ms after last event before refreshing
        self.timer_lock = threading.Lock()
    
    def on_created(self, event):
        """Handle file creation events."""
        if not event.is_directory and self._is_image_file(event.src_path):
            logger.info(f"New image file detected: {event.src_path}")
            self._schedule_batch_refresh()
    
    def on_deleted(self, event):
        """Handle file deletion events."""
        if not event.is_directory and self._is_image_file(event.src_path):
            logger.info(f"Image file deleted: {event.src_path}")
            self._schedule_batch_refresh()
    
    def on_moved(self, event):
        """Handle file move/rename events."""
        if not event.is_directory:
            src_is_image = self._is_image_file(event.src_path)
            dest_is_image = self._is_image_file(event.dest_path)
            
            if src_is_image or dest_is_image:
                logger.info(f"Image file moved/renamed: {event.src_path} -> {event.dest_path}")
                self._schedule_batch_refresh()
    
    def _is_image_file(self, file_path):
        """Check if file is an image based on extension."""
        return file_path.lower().endswith(self.valid_exts)
    
    def _schedule_batch_refresh(self):
        """Schedule a batched refresh that waits for multiple events to settle."""
        import threading
        
        with self.timer_lock:
            # Cancel any existing timer
            if self.refresh_timer is not None:
                self.refresh_timer.cancel()
            
            # Set up a new timer
            self.refresh_timer = threading.Timer(self.batch_delay, self._execute_refresh)
            self.refresh_timer.start()
            self.pending_refresh = True
    
    def _execute_refresh(self):
        """Execute the actual refresh after the batch delay."""
        with self.timer_lock:
            if self.pending_refresh:
                self.pending_refresh = False
                self.refresh_timer = None
                # Schedule refresh in a separate thread to avoid blocking
                import threading
                threading.Thread(target=self.folder_capture._refresh_file_list_async, daemon=True).start()
                logger.info("Executing batched file list refresh")


class FolderCapture(VideoCaptureBase):
    """
    Capture class for reading images from a folder as a video stream.
    Images can be sorted by creation time or by name.
    Supports automatic folder monitoring to detect new/removed images.
    """
    def __init__(self, source: str, sort_by: str = 'name', width: Optional[int] = None, height: Optional[int] = None, fps: float = 30.0, real_time: bool = True, loop: bool = False, watch_folder: bool = True, **kwargs):
        super().__init__(source, **kwargs)
        self.sort_by = sort_by
        self.width = width
        self.height = height
        self.fps = fps
        self.real_time = real_time
        self.loop = loop
        self.watch_folder = watch_folder and WATCHDOG_AVAILABLE
        self.image_files: List[str] = []
        self.index = 0
        self.time_of_last_frame = 0.0
        self._capture_thread = None
        self._stop_event = None
        self._latest_frame = None
        self._latest_success = False
        
        # Folder watching components
        self._folder_observer = None
        self._folder_handler = None
        self._refresh_lock = None
        self._monitor_thread = None
        self._monitor_stop_event = None
        self._last_file_count = 0
        
        if not WATCHDOG_AVAILABLE and watch_folder:
            logger.warning("Folder watching requested but watchdog library not available. Install with 'pip install watchdog'.")
    
    def _start_folder_watching(self):
        """Start monitoring the folder for file changes."""
        if not self.watch_folder or not WATCHDOG_AVAILABLE:
            return
        
        try:
            import threading
            self._refresh_lock = threading.Lock()
            
            # Create and configure the folder watcher
            if WATCHDOG_AVAILABLE:
                from watchdog.observers import Observer
                from watchdog.events import FileSystemEventHandler
                
                class WatchdogHandler(FileSystemEventHandler):
                    def __init__(self, folder_handler):
                        self.folder_handler = folder_handler
                    
                    def on_created(self, event):
                        self.folder_handler.on_created(event)
                    
                    def on_deleted(self, event):
                        self.folder_handler.on_deleted(event)
                    
                    def on_moved(self, event):
                        self.folder_handler.on_moved(event)
                
                self._folder_handler = FolderWatcherHandler(self)
                watchdog_handler = WatchdogHandler(self._folder_handler)
                
                self._folder_observer = Observer()
                self._folder_observer.schedule(watchdog_handler, self.source, recursive=False)
                self._folder_observer.start()
                
                # Start background monitoring thread for additional resilience
                self._start_background_monitor()
                
                logger.info(f"Started folder watching for: {self.source}")
        except Exception as e:
            logger.error(f"Failed to start folder watching: {e}")
            self._folder_observer = None
    
    def _start_background_monitor(self):
        """Start a background thread that periodically checks for file changes."""
        if self._monitor_thread and self._monitor_thread.is_alive():
            return
        
        import threading
        self._monitor_stop_event = threading.Event()
        self._last_file_count = len(self.image_files)
        
        def monitor_loop():
            while not self._monitor_stop_event.is_set():
                try:
                    # Check every 2 seconds for file count changes
                    self._monitor_stop_event.wait(2.0)
                    if self._monitor_stop_event.is_set():
                        break
                    
                    # Quick check: count files in directory
                    if os.path.isdir(self.source):
                        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
                        current_files = [f for f in os.listdir(self.source) 
                                       if f.lower().endswith(valid_exts)]
                        current_count = len(current_files)
                        
                        if current_count != self._last_file_count:
                            logger.info(f"Background monitor detected file count change: {self._last_file_count} -> {current_count}")
                            self._last_file_count = current_count
                            # Trigger refresh
                            threading.Thread(target=self._refresh_file_list_async, daemon=True).start()
                            
                except Exception as e:
                    logger.warning(f"Background monitor error: {e}")
        
        self._monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        self._monitor_thread.start()
        logger.info("Started background file monitor")
    
    def _stop_background_monitor(self):
        """Stop the background monitoring thread."""
        if self._monitor_stop_event:
            self._monitor_stop_event.set()
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1)
        self._monitor_thread = None
        self._monitor_stop_event = None
    
    def _stop_folder_watching(self):
        """Stop monitoring the folder for file changes."""
        # Stop background monitor first
        self._stop_background_monitor()
        
        if self._folder_observer:
            try:
                self._folder_observer.stop()
                self._folder_observer.join(timeout=2)
                logger.info("Stopped folder watching")
            except Exception as e:
                logger.error(f"Error stopping folder watcher: {e}")
            finally:
                self._folder_observer = None
                self._folder_handler = None
    
    def _refresh_file_list_async(self):
        """Async version of refresh_file_list to avoid blocking the file watcher."""
        if hasattr(self, '_refresh_lock') and self._refresh_lock:
            with self._refresh_lock:
                self._refresh_file_list()
        else:
            self._refresh_file_list()

    def start_async(self):
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
        self._last_file_count = len(files)  # Initialize file count tracking
        
        # Start folder watching if enabled
        self._start_folder_watching()
        
        logger.info(f"Connected to folder with {len(files)} images. Folder watching: {'enabled' if self.watch_folder else 'disabled'}")
        return True

    def disconnect(self) -> bool:
        # Stop folder watching first
        self._stop_folder_watching()
        
        self.is_connected = False
        self.image_files = []
        self.index = 0
        logger.info("Disconnected from folder capture.")
        return True

    def _refresh_file_list(self) -> bool:
        """
        Refresh the file list to handle files that may have been added, removed, or renamed.
        
        Returns:
            bool: True if file list was successfully refreshed
        """
        if not os.path.isdir(self.source):
            logger.error(f"Source folder no longer exists: {self.source}")
            return False
        
        # Store current file path WITHOUT calling get_current_file_path() to avoid recursion
        current_file = None
        if self.image_files and 0 <= self.index - 1 < len(self.image_files):
            current_file = self.image_files[self.index - 1] if self.index > 0 else self.image_files[0]
        
        # Re-scan folder for image files
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif')
        files = [os.path.join(self.source, f) for f in os.listdir(self.source) 
                if f.lower().endswith(valid_exts) and os.path.exists(os.path.join(self.source, f))]
        
        if self.sort_by == 'date':
            files.sort(key=lambda x: os.path.getctime(x))
        else:
            files.sort()  # Default: sort by name
        
        if not files:
            logger.warning(f"No image files found in folder after refresh: {self.source}")
            self.image_files = []
            self.index = 0
            return False
        
        # Update file list
        old_count = len(self.image_files)
        self.image_files = files
        new_count = len(self.image_files)
        
        # Update the background monitor's file count tracking
        self._last_file_count = new_count
        
        # Try to maintain current position if the file still exists
        if current_file and current_file in self.image_files:
            self.index = self.image_files.index(current_file) + 1  # +1 because index is incremented after reading
        else:
            # File no longer exists, clamp index to valid range
            self.index = min(self.index, len(self.image_files))
            if self.index <= 0:
                self.index = 0
        
        logger.info(f"Refreshed file list: {old_count} -> {new_count} files, current index: {self.index}")
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
        
        # Check if file still exists before trying to read it
        if not os.path.exists(img_path):
            logger.warning(f"Image file no longer exists: {img_path}")
            # Try to refresh the file list to see if files were moved/renamed
            self._refresh_file_list()
            # Skip this file and try the next one
            self.index += 1
            return False, None
        
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
        logger.warning("Exposure control not applicable for folder capture.")
        return None

    def set_gain(self, value: float) -> bool:
        logger.warning("Gain control not applicable for folder capture.")
        return False

    def get_gain(self) -> Optional[float]:
        logger.warning("Gain control not applicable for folder capture.")
        return None

    def enable_auto_exposure(self, enable: bool = True) -> bool:
        logger.warning("Auto exposure control not applicable for folder capture.")
        return False

    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        if not self.image_files:
            return None
        
        # Find the first existing image file
        for img_path in self.image_files:
            if os.path.exists(img_path):
                img = cv2.imread(img_path)
                if img is not None:
                    h, w = img.shape[:2]
                    if self.width is not None and self.height is not None:
                        return (self.width, self.height)
                    return (w, h)
        
        # If no valid images found, try refreshing the file list
        logger.warning("No valid image files found for frame size detection")
        if self._refresh_file_list() and self.image_files:
            # Try again with refreshed list
            for img_path in self.image_files:
                if os.path.exists(img_path):
                    img = cv2.imread(img_path)
                    if img is not None:
                        h, w = img.shape[:2]
                        if self.width is not None and self.height is not None:
                            return (self.width, self.height)
                        return (w, h)
        
        return None

    def set_frame_size(self, width: int, height: int) -> bool:
        self.width = width
        self.height = height
        return True

    def get_fps(self) -> Optional[float]:
        return self.fps

    def set_fps(self, fps: float) -> bool:
        self.fps = fps
        return True

    def get_current_file_path(self) -> Optional[str]:
        """
        Get the file path of the current image.
        
        Returns:
            Optional[str]: Path to the current image file, or None if no current file
        """
        if not self.is_connected or not self.image_files:
            return None
        
        # Get the current index (accounting for the fact that index is incremented after reading)
        current_index = self.index - 1 if self.index > 0 else 0
        
        # Handle case where we've reached the end and looping is disabled
        if current_index >= len(self.image_files):
            if self.loop:
                current_index = 0
            else:
                # Return the last valid file path
                current_index = len(self.image_files) - 1
        
        # Ensure index is within bounds
        if 0 <= current_index < len(self.image_files):
            file_path = self.image_files[current_index]
            # Check if file still exists
            if os.path.exists(file_path):
                return file_path
            else:
                logger.warning(f"Current file no longer exists: {file_path}")
                # Try to refresh file list
                if self._refresh_file_list():
                    # Retry with refreshed list
                    if 0 <= current_index < len(self.image_files):
                        return self.image_files[current_index]
        
        return None

    def get_current_file_info(self) -> Optional[dict]:
        """
        Get detailed information about the current image file.
        
        Returns:
            Optional[dict]: Dictionary containing file information, or None if no current file
        """
        current_path = self.get_current_file_path()
        if current_path is None:
            return None
        
        try:
            file_stat = os.stat(current_path)
            return {
                'path': current_path,
                'filename': os.path.basename(current_path),
                'size_bytes': file_stat.st_size,
                'creation_time': file_stat.st_ctime,
                'modification_time': file_stat.st_mtime,
                'index': self.index - 1 if self.index > 0 else 0,
                'total_files': len(self.image_files)
            }
        except OSError as e:
            logger.warning(f"Could not get file info for {current_path}: {e}")
            return None

    def get_file_list(self) -> List[str]:
        """
        Get the complete list of image files in the folder.
        
        Returns:
            List[str]: List of all image file paths
        """
        return self.image_files.copy()

    def get_current_index(self) -> int:
        """
        Get the current file index.
        
        Returns:
            int: Current index in the file list
        """
        return self.index - 1 if self.index > 0 else 0

    def set_current_index(self, index: int) -> bool:
        """
        Set the current file index to jump to a specific image.
        
        Args:
            index: Index to jump to (0-based)
            
        Returns:
            bool: True if index was set successfully
        """
        if not self.is_connected or not self.image_files:
            return False
        
        if 0 <= index < len(self.image_files):
            # Check if the target file exists
            target_file = self.image_files[index]
            if os.path.exists(target_file):
                self.index = index
                return True
            else:
                logger.warning(f"Target file at index {index} no longer exists: {target_file}")
                # Try refreshing file list
                if self._refresh_file_list():
                    # Check if index is still valid after refresh
                    if 0 <= index < len(self.image_files):
                        self.index = index
                        return True
                return False
        else:
            logger.warning(f"Index {index} out of range (0-{len(self.image_files)-1})")
            return False

    def validate_file_list(self) -> int:
        """
        Validate all files in the list and remove any that no longer exist.
        
        Returns:
            int: Number of files removed
        """
        if not self.image_files:
            return 0
        
        original_count = len(self.image_files)
        current_file = self.get_current_file_path()
        
        # Filter out files that no longer exist
        valid_files = [f for f in self.image_files if os.path.exists(f)]
        
        removed_count = original_count - len(valid_files)
        
        if removed_count > 0:
            logger.warning(f"Removed {removed_count} missing files from list")
            self.image_files = valid_files
            
            # Adjust current index
            if current_file and current_file in self.image_files:
                self.index = self.image_files.index(current_file)
            else:
                # Current file was removed, clamp index
                self.index = min(self.index, len(self.image_files) - 1)
                if self.index < 0:
                    self.index = 0
        
        return removed_count

    def get_missing_files(self) -> List[str]:
        """
        Get a list of files that are in the file list but no longer exist on disk.
        
        Returns:
            List[str]: List of missing file paths
        """
        return [f for f in self.image_files if not os.path.exists(f)]

    def enable_folder_watching(self, enable: bool = True) -> bool:
        """
        Enable or disable folder watching.
        
        Args:
            enable: True to enable, False to disable
            
        Returns:
            bool: True if successful
        """
        if not WATCHDOG_AVAILABLE:
            logger.warning("Folder watching not available - watchdog library not installed")
            return False
        
        if enable and not self.watch_folder:
            self.watch_folder = True
            if self.is_connected:
                self._start_folder_watching()
            return True
        elif not enable and self.watch_folder:
            self.watch_folder = False
            self._stop_folder_watching()
            return True
        
        return True  # Already in desired state

    def is_folder_watching_enabled(self) -> bool:
        """
        Check if folder watching is currently enabled.
        
        Returns:
            bool: True if folder watching is enabled
        """
        return self.watch_folder and self._folder_observer is not None

    def get_folder_watching_status(self) -> dict:
        """
        Get detailed folder watching status.
        
        Returns:
            dict: Status information
        """
        return {
            'watchdog_available': WATCHDOG_AVAILABLE,
            'watching_enabled': self.watch_folder,
            'observer_running': self._folder_observer is not None and self._folder_observer.is_alive() if self._folder_observer else False,
            'folder_path': self.source,
            'total_files': len(self.image_files),
            'missing_files': len(self.get_missing_files())
        }

    def force_refresh(self) -> int:
        """
        Manually trigger a file list refresh.
        
        Returns:
            int: Number of files in the updated list
        """
        self._refresh_file_list()
        return len(self.image_files)

# Standalone test code
if __name__ == "__main__":
    import sys
    
    # Use command line argument for folder path, or default to a test folder
    if len(sys.argv) > 1:
        folder = sys.argv[1]
    else:
        # Try some common test folders
        test_folders = [
            "C:/Users/olive/OneDrive/Desktop/image_folder",
            "./media/image_seq",  # Relative to project root
            "../media/image_seq",  # If running from frame_source directory
            "../../media/image_seq",  # If running from deeper nested folder
            "./test_images",
            "../test_images"
        ]
        
        folder = None
        for test_folder in test_folders:
            if os.path.exists(test_folder) and os.path.isdir(test_folder):
                folder = test_folder
                break
        
        if folder is None:
            print("No test folder found. Usage: python folder_capture.py <folder_path>")
            print(f"Tried folders: {test_folders}")
            sys.exit(1)
    
    print(f"Testing FolderCapture with folder: {folder}")
    
    cap = FolderCapture(folder, sort_by='date', fps=10, real_time=True, loop=True, watch_folder=True)
    if cap.connect():
        print(f"Connected successfully! Found {len(cap.get_file_list())} image files.")
        
        # Show folder watching status
        watch_status = cap.get_folder_watching_status()
        print(f"Folder watching status: {watch_status}")
        
        # cap.start()
        cv2.namedWindow("FolderCapture", cv2.WINDOW_NORMAL)
        frame_count = 0
        last_validation_time = time.time()
        last_file_count = len(cap.get_file_list())
        
        # Check for missing files initially
        missing_files = cap.get_missing_files()
        if missing_files:
            print(f"Warning: {len(missing_files)} files are missing from the start")
        
        print("Press 'q' to quit, 's' to skip to next image, 'r' to refresh file list, 'w' to toggle folder watching")
        
        while cap.is_connected:
            ret, frame = cap.read()
            if ret:
                frame_count += 1
                if frame is not None:
                    # Display current file information
                    current_file = cap.get_current_file_path()
                    file_info = cap.get_current_file_info()
                    
                    if current_file:
                        print(f"Frame {frame_count}: {os.path.basename(current_file)}")
                        if file_info:
                            print(f"  Index: {file_info['index']}/{file_info['total_files']-1}")
                            print(f"  Size: {file_info['size_bytes']} bytes")
                    
                    # Check if file count changed (new files detected)
                    current_file_count = len(cap.get_file_list())
                    if current_file_count != last_file_count:
                        print(f"File count changed: {last_file_count} -> {current_file_count}")
                        last_file_count = current_file_count
                    
                    # Periodically validate file list (every 30 seconds)
                    current_time = time.time()
                    if current_time - last_validation_time > 30:
                        removed_count = cap.validate_file_list()
                        if removed_count > 0:
                            print(f"Validation: Removed {removed_count} missing files")
                        last_validation_time = current_time
                    
                    cv2.imshow("FolderCapture", frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quitting...")
                    break
                elif key == ord('s'):
                    print("Skipping to next image...")
                    # Force skip by incrementing index
                    current_idx = cap.get_current_index()
                    cap.set_current_index(current_idx + 1)
                elif key == ord('r'):
                    print("Refreshing file list...")
                    new_count = cap.force_refresh()
                    print(f"Refreshed: now have {new_count} files")
                elif key == ord('w'):
                    # Toggle folder watching
                    current_status = cap.is_folder_watching_enabled()
                    cap.enable_folder_watching(not current_status)
                    new_status = cap.is_folder_watching_enabled()
                    print(f"Folder watching {'enabled' if new_status else 'disabled'}")
                    
            else:
                print("No more frames available")
                if not cap.loop:
                    print("Reached end of image sequence (loop=False)")
                    break
                    
        # cap.stop()
        cap.disconnect()
        cv2.destroyAllWindows()
        print("FolderCapture test completed.")
    else:
        print(f"Failed to connect to folder: {folder}")
        print("Make sure the folder exists and contains image files (.jpg, .jpeg, .png, .bmp, .tiff, .tif)")
