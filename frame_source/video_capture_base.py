import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any

class VideoCaptureBase(ABC):
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
        Internal: Calls the subclass's read() method for background thread use.
        Override this if you need special thread-safety in subclasses.
        """
        return self.read()

    def get_latest_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the most recent frame captured by the background thread.
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        return getattr(self, '_latest_success', False), getattr(self, '_latest_frame', None)

    """Abstract base class for video capture implementations."""
    
    def __init__(self, source: Any = None, **kwargs):
        """
        Initialize the capture device.
        
        Args:
            source: Source identifier (device index, URL, etc.)
            **kwargs: Additional parameters specific to the capture type
        """
        self.source = source
        self.is_connected = False
        self._exposure = None
        self._gain = None
        self.config = kwargs
        self.type = self.__class__.__name__
        
    def __str__(self) -> str:
        return f"{self.type}(source={self.source}, connected={self.is_connected})"

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the capture device.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def disconnect(self) -> bool:
        """
        Disconnect from the capture device.
        
        Returns:
            bool: True if disconnection successful, False otherwise
        """
        pass
    
    @abstractmethod
    def _read_implementation(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a frame from the capture device.
        If background capture is running, returns the latest frame.
        Otherwise, reads a new frame from the device.
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        pass

    @abstractmethod
    def enable_auto_exposure(self, enable: bool = True) -> bool:
        """
        Enable or disable auto exposure.
        
        Args:
            enable: True to enable, False to disable
            
        Returns:
            bool: True if set successfully, False otherwise
        """
        pass
    

    def get_exposure_range(self) -> Optional[Tuple[float, float]]:
        """
        Get the minimum and maximum exposure values supported by the device.
        Returns:
            Optional[Tuple[float, float]]: (min_exposure, max_exposure) or None if not available
        """
        return None

    def get_gain_range(self) -> Optional[Tuple[float, float]]:
        """
        Get the minimum and maximum gain values supported by the device.
        Returns:
            Optional[Tuple[float, float]]: (min_gain, max_gain) or None if not available
        """
        return None
    

    @abstractmethod
    def set_exposure(self, value: float) -> bool:
        """
        Set exposure value.
        
        Args:
            value: Exposure value (range depends on implementation)
            
        Returns:
            bool: True if set successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_exposure(self) -> Optional[float]:
        """
        Get current exposure value.
        
        Returns:
            Optional[float]: Current exposure value or None if not available
        """
        pass
    
    @abstractmethod
    def set_gain(self, value: float) -> bool:
        """
        Set gain value.
        
        Args:
            value: Gain value (range depends on implementation)
            
        Returns:
            bool: True if set successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def get_gain(self) -> Optional[float]:
        """
        Get current gain value.
        
        Returns:
            Optional[float]: Current gain value or None if not available
        """
        pass
    
    @classmethod
    def discover(cls) -> list:
        """
        Discover available devices for this capture type.
        
        Returns:
            list: List of available devices. Format depends on implementation:
                - For cameras: List of device indices, serial numbers, or device info dicts
                - For audio: List of microphone devices with info
                - For network cameras: List of discovered IP cameras
                - For files: Empty list (no discovery needed)
        
        Note:
            This is a class method that can be called without instantiating the capture class.
            Subclasses should override this method to implement device-specific discovery.
            Default implementation returns empty list.
        """
        return []
    
    def get_frame_size(self) -> Optional[Tuple[int, int]]:
        """
        Get frame dimensions (width, height).
        
        Returns:
            Optional[Tuple[int, int]]: Frame size or None if not available
        """
        return None
    
    def set_frame_size(self, width: int, height: int) -> bool:
        """
        Set frame dimensions.
        
        Args:
            width: Frame width
            height: Frame height
            
        Returns:
            bool: True if set successfully, False otherwise
        """
        return False
    
    def get_fps(self) -> Optional[float]:
        """Get current FPS."""
        return None
    
    def set_fps(self, fps: float) -> bool:
        """Set FPS."""
        return False
    
    def __enter__(self):
        """Context manager entry."""
        if not self.is_connected:
            self.connect()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        if self.is_connected:
            self.disconnect()
    
    def isOpened(self) -> bool:
        """
        Check if the capture device is opened/connected.
        OpenCV-compatible API method.
        
        Returns:
            bool: True if device is connected, False otherwise
        """
        return self.is_connected
    
    def release(self) -> None:
        """
        Release the capture device and stop any background threads.
        OpenCV-compatible API method that combines disconnect and stop.
        
        This method ensures clean shutdown by:
        1. Stopping background capture threads if running
        2. Disconnecting from the capture device
        """
        # Stop background threads first
        if hasattr(self, '_capture_thread') and self._capture_thread:
            self.stop()
        
        # Then disconnect from the device
        if self.is_connected:
            self.disconnect()
            
    def attach_processor(self, processor):
        """Attach a frame processor to this camera."""
        if not hasattr(self, '_processors'):
            self._processors = []
        self._processors.append(processor)
        return processor
    
    def detach_processor(self, processor):
        """Remove a processor from this camera."""
        if hasattr(self, '_processors'):
            if processor in self._processors:
                self._processors.remove(processor)
                return True
        return False
    
    def clear_processors(self):
        """Remove all processors."""
        if hasattr(self, '_processors'):
            self._processors.clear()

    def get_processors(self):
        """Get all attached processors."""
        if not hasattr(self, '_processors'):
            self._processors = []
        return self._processors

    def read(self):
        """Read a frame from the camera and apply any attached processors."""
        ret, frame = self._read_implementation()  # Your existing read logic
        
        if ret and frame is not None and hasattr(self, '_processors') and self._processors:
            for processor in self._processors:
                frame = processor.process(frame)
        
        return ret, frame
