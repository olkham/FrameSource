import numpy as np
import cv2
import time
from typing import Optional, Tuple, Any, Dict
import logging
import threading

try:
    from .video_capture_base import VideoCaptureBase
except ImportError:
    # If running as main script, try absolute import
    from video_capture_base import VideoCaptureBase

try:
    import mss
except ImportError:
    mss = None
    logging.warning("mss is not installed. Install it with 'pip install mss' to use ScreenCapture.")

import platform
SYSTEM = platform.system()

# Windows support
try:
    import win32gui
    import win32con
    WINDOWS_AVAILABLE = True
except ImportError:
    win32gui = None
    win32con = None
    WINDOWS_AVAILABLE = False
    if SYSTEM == 'Windows':
        logging.warning("pywin32 is not installed. Install it with 'pip install pywin32' to discover windows.")

# macOS support
try:
    if SYSTEM == 'Darwin':
        from Quartz import (
            CGWindowListCopyWindowInfo,
            kCGWindowListOptionOnScreenOnly,
            kCGNullWindowID,
            kCGWindowLayer,
            kCGWindowName,
            kCGWindowOwnerName,
            kCGWindowBounds,
            kCGWindowNumber
        )
        MACOS_AVAILABLE = True
    else:
        MACOS_AVAILABLE = False
except ImportError:
    MACOS_AVAILABLE = False
    if SYSTEM == 'Darwin':
        logging.warning("pyobjc-framework-Quartz is not installed. Install it with 'pip install pyobjc-framework-Quartz' to discover windows on macOS.")

# Linux support
try:
    if SYSTEM == 'Linux':
        import subprocess
        LINUX_AVAILABLE = True
    else:
        LINUX_AVAILABLE = False
except ImportError:
    LINUX_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ScreenCapture(VideoCaptureBase):
    """
    Capture class for grabbing frames from a region of the screen.
    Args:
        x (int): Top-left x coordinate
        y (int): Top-left y coordinate
        w (int): Width of region
        h (int): Height of region
        fps (float): Target FPS (default 30)
    """
    has_discovery = True
    display_fields = [
        {'key': 'name', 'label': 'Name'},
        {'key': 'type', 'label': 'Type'},
        {'key': 'title', 'label': 'Title'},
        {'key': 'width', 'label': 'Width'},
        {'key': 'height', 'label': 'Height'},
        {'key': 'left', 'label': 'X'},
        {'key': 'top', 'label': 'Y'}
    ]
    
    def __init__(self, x: int = 0, y: int = 0, w: int = 640, h: int = 480, fps: float = 30.0, **kwargs):
        super().__init__(**kwargs)
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.fps = fps
        self.monitor = {"top": y, "left": x, "width": w, "height": h}
        self._thread_local = threading.local()
        self.is_connected = False
        self.time_of_last_frame = 0.0
        self.hwnd = kwargs.get('hwnd', None)  # Optional window handle for tracking
    
    @classmethod
    def from_window(cls, window_id: Any, fps: float = 30.0, **kwargs):
        """
        Create a ScreenCapture instance configured to capture a specific window.
        
        Args:
            window_id: Window identifier (hwnd on Windows, window number on macOS, window ID on Linux)
            fps: Target frame rate
            **kwargs: Additional parameters
        
        Returns:
            ScreenCapture: Configured instance
        """
        if SYSTEM == 'Windows':
            if not WINDOWS_AVAILABLE:
                raise RuntimeError("Window capture requires pywin32. Install with 'pip install pywin32'")
            
            try:
                rect = win32gui.GetWindowRect(window_id)
                left, top, right, bottom = rect
                width = right - left
                height = bottom - top
                
                instance = cls(x=left, y=top, w=width, h=height, fps=fps, hwnd=window_id, **kwargs)
                return instance
                
            except Exception as e:
                raise RuntimeError(f"Failed to get window dimensions for hwnd {window_id}: {e}")
        
        elif SYSTEM == 'Darwin':
            if not MACOS_AVAILABLE:
                raise RuntimeError("Window capture requires pyobjc-framework-Quartz. Install with 'pip install pyobjc-framework-Quartz'")
            
            try:
                # Get window info
                window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
                for window in window_list:
                    if window.get(kCGWindowNumber) == window_id:
                        bounds = window.get(kCGWindowBounds)
                        left = int(bounds['X'])
                        top = int(bounds['Y'])
                        width = int(bounds['Width'])
                        height = int(bounds['Height'])
                        
                        instance = cls(x=left, y=top, w=width, h=height, fps=fps, hwnd=window_id, **kwargs)
                        return instance
                
                raise RuntimeError(f"Window {window_id} not found")
                
            except Exception as e:
                raise RuntimeError(f"Failed to get window dimensions for window {window_id}: {e}")
        
        elif SYSTEM == 'Linux':
            if not LINUX_AVAILABLE:
                raise RuntimeError("Window capture requires wmctrl or xdotool")
            
            try:
                # Try to get window geometry using xdotool
                result = subprocess.run(
                    ['xdotool', 'getwindowgeometry', '--shell', str(window_id)],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                # Parse output
                geometry = {}
                for line in result.stdout.strip().split('\n'):
                    if '=' in line:
                        key, value = line.split('=', 1)
                        geometry[key] = int(value)
                
                left = geometry.get('X', 0)
                top = geometry.get('Y', 0)
                width = geometry.get('WIDTH', 640)
                height = geometry.get('HEIGHT', 480)
                
                instance = cls(x=left, y=top, w=width, h=height, fps=fps, hwnd=window_id, **kwargs)
                return instance
                
            except subprocess.CalledProcessError:
                # Try wmctrl as fallback
                try:
                    result = subprocess.run(
                        ['wmctrl', '-lG'],
                        capture_output=True,
                        text=True,
                        check=True
                    )
                    
                    for line in result.stdout.strip().split('\n'):
                        parts = line.split(None, 7)
                        if len(parts) >= 7:
                            win_id = parts[0]
                            if win_id == hex(window_id) or win_id == str(window_id):
                                left = int(parts[2])
                                top = int(parts[3])
                                width = int(parts[4])
                                height = int(parts[5])
                                
                                instance = cls(x=left, y=top, w=width, h=height, fps=fps, hwnd=window_id, **kwargs)
                                return instance
                    
                    raise RuntimeError(f"Window {window_id} not found in wmctrl output")
                    
                except Exception as e:
                    raise RuntimeError(f"Failed to get window dimensions using wmctrl: {e}")
            
            except Exception as e:
                raise RuntimeError(f"Failed to get window dimensions for window {window_id}: {e}")
        
        else:
            raise RuntimeError(f"Window capture not supported on {SYSTEM}")

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
        Internal: Calls the direct read method for background thread use (avoids recursion).
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
        if mss is None:
            logger.error("mss is not installed. Cannot use ScreenCapture.")
            return False
        self.is_connected = True
        logger.info(f"ScreenCapture connected to region x={self.x}, y={self.y}, w={self.w}, h={self.h}")
        return True

    def disconnect(self) -> bool:
        self.is_connected = False
        logger.info("ScreenCapture disconnected.")
        return True

    def _read_implementation(self) -> Tuple[bool, Optional[np.ndarray]]:
        # If background thread is running, return latest frame
        if hasattr(self, '_capture_thread') and self._capture_thread is not None and self._capture_thread.is_alive():
            return self.get_latest_frame()
        else:
            return self._read_direct()

    def _get_sct(self):
        if not hasattr(self._thread_local, 'sct') or self._thread_local.sct is None:
            if mss is None:
                raise RuntimeError("mss is not installed. Cannot create screen capture.")
            self._thread_local.sct = mss.mss()
        return self._thread_local.sct

    def _read_direct(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_connected:
            return False, None
        sct = self._get_sct()
        # Real-time playback control
        if self.fps > 0:
            frame_duration = 1.0 / self.fps
            now = time.time()
            elapsed = now - self.time_of_last_frame
            if elapsed < frame_duration:
                time.sleep(frame_duration - elapsed)
            self.time_of_last_frame = time.time()
        img = np.array(sct.grab(self.monitor))
        # Convert BGRA to BGR
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        return True, frame

    def set_exposure(self, value: float) -> bool:
        logger.warning("Exposure control not applicable for screen capture.")
        return False

    def get_exposure(self) -> Optional[float]:
        return None

    def set_gain(self, value: float) -> bool:
        logger.warning("Gain control not applicable for screen capture.")
        return False

    def get_gain(self) -> Optional[float]:
        return None

    def enable_auto_exposure(self, enable: bool = True) -> bool:
        logger.warning("Auto exposure control not applicable for screen capture.")
        return False

    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        return (self.w, self.h)

    def set_frame_size(self, width: int, height: int) -> bool:
        self.w = width
        self.h = height
        self.monitor["width"] = width
        self.monitor["height"] = height
        return True

    def get_fps(self) -> Optional[float]:
        return self.fps

    def set_fps(self, fps: float) -> bool:
        self.fps = fps
        return True

    @classmethod
    def discover(cls) -> list:
        """
        Discover available screen capture sources (monitors/displays and windows).
        
        Returns:
            list: List of dictionaries containing screen/window information.
                Each dict contains: 
                - For monitors: {'type': 'monitor', 'index': int, 'name': str, 'width': int, 'height': int, 'left': int, 'top': int}
                - For windows: {'type': 'window', 'id': str/int, 'name': str, 'title': str, 'width': int, 'height': int, 'left': int, 'top': int, 'is_visible': bool}
        """
        devices = []
        
        # Discover monitors
        if mss is None:
            logger.warning("mss module not available. Cannot discover screen sources.")
        else:
            try:
                with mss.mss() as sct:
                    # Get all monitors (index 0 is typically all monitors combined)
                    monitors = sct.monitors
                    
                    for i, monitor in enumerate(monitors):
                        device_data = {
                            'type': 'monitor',
                            'index': i,
                            'id': f"monitor_{i}",
                            'name': f"Monitor {i}" if i > 0 else "All Monitors",
                            'width': monitor['width'],
                            'height': monitor['height'],
                            'left': monitor['left'],
                            'top': monitor['top']
                        }
                        devices.append(device_data)
                        logger.info(f"Found screen source: {device_data}")
                        
            except Exception as e:
                logger.error(f"Error discovering screen sources: {e}")
        
        # Discover windows - platform specific
        if SYSTEM == 'Windows':
            devices.extend(cls._discover_windows_windows())
        elif SYSTEM == 'Darwin':
            devices.extend(cls._discover_windows_macos())
        elif SYSTEM == 'Linux':
            devices.extend(cls._discover_windows_linux())
        else:
            logger.info(f"Window discovery not supported on {SYSTEM}")
        
        return devices
    
    @classmethod
    def _discover_windows_windows(cls) -> list:
        """Discover windows on Windows using win32gui."""
        if not WINDOWS_AVAILABLE:
            logger.info("Window discovery not available (pywin32 not installed)")
            return []
        
        windows = []
        
        try:
            def window_callback(hwnd, extra):
                """Callback function to enumerate windows"""
                if not win32gui.IsWindowVisible(hwnd):
                    return
                
                # Get window title
                title = win32gui.GetWindowText(hwnd)
                
                # Skip windows without titles or with empty titles
                if not title or len(title.strip()) == 0:
                    return
                
                # Get window rectangle
                try:
                    rect = win32gui.GetWindowRect(hwnd)
                    left, top, right, bottom = rect
                    width = right - left
                    height = bottom - top
                    
                    # Skip windows that are too small (likely not real windows)
                    if width < 50 or height < 50:
                        return
                    
                    # Get class name for additional context
                    try:
                        class_name = win32gui.GetClassName(hwnd)
                    except:
                        class_name = "Unknown"
                    
                    window_data = {
                        'type': 'window',
                        'hwnd': hwnd,
                        'id': f"window_{hwnd}",
                        'name': title,
                        'title': title,
                        'class_name': class_name,
                        'width': width,
                        'height': height,
                        'left': left,
                        'top': top,
                        'is_visible': True
                    }
                    windows.append(window_data)
                    
                except Exception as e:
                    logger.debug(f"Error getting window rect for hwnd {hwnd}: {e}")
            
            # Enumerate all windows
            win32gui.EnumWindows(window_callback, None)
            
            # Sort windows by title for consistent ordering
            windows.sort(key=lambda w: w['title'].lower())
            
            logger.info(f"Found {len(windows)} visible windows")
            
        except Exception as e:
            logger.error(f"Error discovering windows: {e}")
        
        return windows
    
    @classmethod
    def _discover_windows_macos(cls) -> list:
        """Discover windows on macOS using Quartz."""
        if not MACOS_AVAILABLE:
            logger.info("Window discovery not available (pyobjc-framework-Quartz not installed)")
            return []
        
        windows = []
        
        try:
            # Get list of all on-screen windows
            window_list = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
            
            for window in window_list:
                # Get window properties
                window_layer = window.get(kCGWindowLayer, 0)
                
                # Skip desktop and other background windows (layer > 0)
                if window_layer != 0:
                    continue
                
                window_name = window.get(kCGWindowName, '')
                owner_name = window.get(kCGWindowOwnerName, '')
                window_number = window.get(kCGWindowNumber, 0)
                bounds = window.get(kCGWindowBounds, {})
                
                # Skip windows without names
                if not window_name or len(window_name.strip()) == 0:
                    continue
                
                # Get dimensions
                left = int(bounds.get('X', 0))
                top = int(bounds.get('Y', 0))
                width = int(bounds.get('Width', 0))
                height = int(bounds.get('Height', 0))
                
                # Skip windows that are too small
                if width < 50 or height < 50:
                    continue
                
                window_data = {
                    'type': 'window',
                    'window_number': window_number,
                    'id': f"window_{window_number}",
                    'name': window_name,
                    'title': window_name,
                    'owner': owner_name,
                    'width': width,
                    'height': height,
                    'left': left,
                    'top': top,
                    'is_visible': True
                }
                windows.append(window_data)
            
            # Sort windows by title for consistent ordering
            windows.sort(key=lambda w: w['title'].lower())
            
            logger.info(f"Found {len(windows)} visible windows")
            
        except Exception as e:
            logger.error(f"Error discovering windows on macOS: {e}")
        
        return windows
    
    @classmethod
    def _discover_windows_linux(cls) -> list:
        """Discover windows on Linux using wmctrl or xdotool."""
        if not LINUX_AVAILABLE:
            return []
        
        windows = []
        
        # Try wmctrl first (more reliable)
        try:
            result = subprocess.run(
                ['wmctrl', '-lGp'],
                capture_output=True,
                text=True,
                check=True
            )
            
            for line in result.stdout.strip().split('\n'):
                parts = line.split(None, 8)
                if len(parts) >= 9:
                    win_id = parts[0]
                    desktop = parts[1]
                    pid = parts[2]
                    left = int(parts[3])
                    top = int(parts[4])
                    width = int(parts[5])
                    height = int(parts[6])
                    hostname = parts[7]
                    title = parts[8] if len(parts) > 8 else ""
                    
                    # Skip windows without titles
                    if not title or len(title.strip()) == 0:
                        continue
                    
                    # Skip windows that are too small
                    if width < 50 or height < 50:
                        continue
                    
                    # Convert hex window ID to int
                    try:
                        window_id_int = int(win_id, 16)
                    except:
                        window_id_int = win_id
                    
                    window_data = {
                        'type': 'window',
                        'window_id': window_id_int,
                        'id': f"window_{win_id}",
                        'name': title,
                        'title': title,
                        'desktop': desktop,
                        'pid': pid,
                        'width': width,
                        'height': height,
                        'left': left,
                        'top': top,
                        'is_visible': True
                    }
                    windows.append(window_data)
            
            # Sort windows by title for consistent ordering
            windows.sort(key=lambda w: w['title'].lower())
            
            logger.info(f"Found {len(windows)} visible windows using wmctrl")
            
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Try xdotool as fallback
            try:
                result = subprocess.run(
                    ['xdotool', 'search', '--onlyvisible', '--name', '.*'],
                    capture_output=True,
                    text=True,
                    check=True
                )
                
                window_ids = result.stdout.strip().split('\n')
                
                for win_id in window_ids:
                    if not win_id:
                        continue
                    
                    try:
                        # Get window name
                        name_result = subprocess.run(
                            ['xdotool', 'getwindowname', win_id],
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        title = name_result.stdout.strip()
                        
                        if not title or len(title.strip()) == 0:
                            continue
                        
                        # Get window geometry
                        geom_result = subprocess.run(
                            ['xdotool', 'getwindowgeometry', '--shell', win_id],
                            capture_output=True,
                            text=True,
                            check=True
                        )
                        
                        # Parse geometry
                        geometry = {}
                        for line in geom_result.stdout.strip().split('\n'):
                            if '=' in line:
                                key, value = line.split('=', 1)
                                try:
                                    geometry[key] = int(value)
                                except:
                                    geometry[key] = value
                        
                        left = geometry.get('X', 0)
                        top = geometry.get('Y', 0)
                        width = geometry.get('WIDTH', 0)
                        height = geometry.get('HEIGHT', 0)
                        
                        # Skip windows that are too small
                        if width < 50 or height < 50:
                            continue
                        
                        window_data = {
                            'type': 'window',
                            'window_id': int(win_id),
                            'id': f"window_{win_id}",
                            'name': title,
                            'title': title,
                            'width': width,
                            'height': height,
                            'left': left,
                            'top': top,
                            'is_visible': True
                        }
                        windows.append(window_data)
                        
                    except Exception as e:
                        logger.debug(f"Error getting info for window {win_id}: {e}")
                        continue
                
                # Sort windows by title for consistent ordering
                windows.sort(key=lambda w: w['title'].lower())
                
                logger.info(f"Found {len(windows)} visible windows using xdotool")
                
            except (subprocess.CalledProcessError, FileNotFoundError):
                logger.warning("Neither wmctrl nor xdotool available. Install one to discover windows on Linux.")
        
        return windows

    @classmethod
    def get_config_schema(cls) -> Dict[str, Any]:
        """Get configuration schema for screen capture"""
        return {
            'title': 'Screen Capture Configuration',
            'description': 'Configure screen/monitor capture settings',
            'fields': [
                {
                    'name': 'x',
                    'label': 'X Position',
                    'type': 'number',
                    'min': 0,
                    'max': 5000,
                    'placeholder': '0',
                    'description': 'Left edge of capture area (pixels from left)',
                    'required': False,
                    'default': 0
                },
                {
                    'name': 'y',
                    'label': 'Y Position',
                    'type': 'number',
                    'min': 0,
                    'max': 5000,
                    'placeholder': '0',
                    'description': 'Top edge of capture area (pixels from top)',
                    'required': False,
                    'default': 0
                },
                {
                    'name': 'w',
                    'label': 'Width',
                    'type': 'number',
                    'min': 160,
                    'max': 5000,
                    'placeholder': '640',
                    'description': 'Width of capture area in pixels',
                    'required': False,
                    'default': 640
                },
                {
                    'name': 'h',
                    'label': 'Height',
                    'type': 'number',
                    'min': 120,
                    'max': 5000,
                    'placeholder': '480',
                    'description': 'Height of capture area in pixels',
                    'required': False,
                    'default': 480
                },
                {
                    'name': 'fps',
                    'label': 'Frame Rate (FPS)',
                    'type': 'number',
                    'min': 1,
                    'max': 60,
                    'placeholder': '30',
                    'description': 'Frames per second for screen capture',
                    'required': False,
                    'default': 30
                }
            ]
        }


if __name__ == "__main__":
    # Example usage
    
    sources = ScreenCapture.discover()
    print(f"\nDiscovered {len(sources)} screen capture sources:\n")
    
    # Display monitors
    monitors = [s for s in sources if s.get('type') == 'monitor']
    if monitors:
        print(f"Monitors ({len(monitors)}):")
        for screen in monitors:
            print(f"  - {screen['name']} (#{screen['index']}): {screen['width']}x{screen['height']} at ({screen['left']}, {screen['top']})")
    
    # Display windows
    windows = [s for s in sources if s.get('type') == 'window']
    if windows:
        print(f"\nWindows ({len(windows)}):")
        for i, window in enumerate(windows[:20]):  # Show first 20 windows
            print(f"  - {window['title'][:60]:60s} | {window['width']}x{window['height']} at ({window['left']}, {window['top']})")
        if len(windows) > 20:
            print(f"  ... and {len(windows) - 20} more windows")
    
    print("\n" + "="*80)
    print("Example: Capture from a specific monitor")
    print("="*80)
    
    camera = ScreenCapture(x=100, y=100, w=800, h=600, fps=30)
    if camera.connect():
        camera.start_async()
        print("Screen capture connected successfully.")
        print(f"Frame size: {camera.get_frame_size()}")
        print(f"FPS: {camera.get_fps()}")
        
        # Read a few frames
        frame_count = 0
        while camera.is_connected and frame_count < 100:
            ret, frame = camera.read()
            if ret:
                cv2.imshow("Screen Capture", frame) # type: ignore
                frame_count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        camera.stop()
        camera.disconnect()
    else:
        print("Failed to connect to screen capture.")