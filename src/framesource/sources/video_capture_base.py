import time
import uuid as _uuid
from collections import deque

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Any, Protocol, runtime_checkable


class Frame(np.ndarray):
    """A numpy array subclass that carries per-frame metadata.

    Because ``Frame`` *is* a numpy array it works transparently with any
    OpenCV or numpy function::

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # works as-is
        print(frame.timestamp, frame.uuid)               # metadata intact

    Attributes:
        timestamp (float): UNIX wall-clock time of capture (``time.time()``).
            Use this for human-readable timestamps, logging, or correlating
            frames with external wall-clock events. It is subject to system
            clock adjustments (NTP sync, manual changes, DST), so it is not
            reliable for measuring elapsed time or latency.
        monotonic (float): Monotonic clock reading at capture
            (``time.monotonic()``). Use this for latency and FPS math
            (elapsed-time computations) since it always moves forward and is
            immune to wall-clock jumps.
        count (int | None): Monotonically-increasing counter per source.
        uuid (str): UUID4 string uniquely identifying this frame.
        source (str | None): Identifies the capture source.
        metadata (dict): Free-form key/value store for extra fields.
    """

    def __new__(
        cls,
        input_array: np.ndarray,
        timestamp: Optional[float] = None,
        count: Optional[int] = None,
        uuid: Optional[str] = None,
        source: Optional[str] = None,
        metadata: Optional[dict] = None,
        monotonic: Optional[float] = None,
    ) -> "Frame":
        obj = np.asarray(input_array).view(cls)
        obj.timestamp = timestamp if timestamp is not None else time.time()
        obj.count = count
        obj.uuid = uuid if uuid is not None else str(_uuid.uuid4())
        obj.source = source
        obj.metadata = metadata if metadata is not None else {}
        obj.monotonic = monotonic if monotonic is not None else time.monotonic()
        return obj

    def __array_finalize__(self, obj: Optional[np.ndarray]) -> None:
        """Called whenever a new view / copy of a Frame is produced."""
        if obj is None:
            return
        self.timestamp = getattr(obj, "timestamp", None)
        self.count = getattr(obj, "count", None)
        self.uuid = getattr(obj, "uuid", None)
        self.source = getattr(obj, "source", None)
        self.metadata = getattr(obj, "metadata", {})
        self.monotonic = getattr(obj, "monotonic", None)

    def __reduce__(self):
        """Extend numpy pickling so metadata survives pickle / deepcopy."""
        pickled = super().__reduce__()
        extra = (self.timestamp, self.count, self.uuid, self.source, self.metadata, self.monotonic)
        np_state = pickled[2] if isinstance(pickled[2], tuple) else (pickled[2],)
        return (pickled[0], pickled[1], np_state + extra)

    def __setstate__(self, state):
        *np_state, ts, cnt, uid, src, meta, mono = state
        super().__setstate__(tuple(np_state))
        self.timestamp = ts
        self.count = cnt
        self.uuid = uid
        self.source = src
        self.metadata = meta
        self.monotonic = mono

class VideoCaptureBase(ABC):
    # Class attribute indicating if this capture type supports device discovery
    has_discovery = False

    # Capability flags. These are base-class defaults; subclasses that lack a
    # given capability (e.g. a source with no controllable exposure/gain, or
    # a depth-producing device) should override them. `supports_discovery`
    # is derived below from `has_discovery` rather than duplicated.
    supports_exposure: bool = True
    supports_gain: bool = True
    supports_depth: bool = False

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
        self._frame_count = 0
        self.config = kwargs
        self.type = self.__class__.__name__
        # Rolling window of recent frame timestamps for fps_actual()
        self._frame_timestamps: deque = deque(maxlen=30)
        
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
        Read a single frame from the capture device.

        This is the synchronous, blocking primitive each subclass must
        implement. Callers should use :meth:`read` (which wraps the result in
        a :class:`Frame` and applies any attached processors) rather than
        calling this directly.

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

    def is_open(self) -> bool:
        """Check if the capture device is connected (Pythonic alias for isOpened)."""
        return self.is_connected

    @property
    def supports_discovery(self) -> bool:
        """Whether this capture type supports :meth:`discover`.

        Read-only alias for the ``has_discovery`` class attribute, exposed as
        a property so it lines up with the other ``supports_*`` capability
        flags (``supports_exposure``, ``supports_gain``, ``supports_depth``).
        """
        return bool(type(self).has_discovery)

    def release(self) -> None:
        """
        Release the capture device.
        OpenCV-compatible API method that disconnects from the device.
        """
        if self.is_connected:
            self.disconnect()
            
    def set(self, prop_id: int, value: Any) -> bool:
        """
        Set a property of the capture device.
        OpenCV-compatible API method.
        
        Args:
            prop_id: Property identifier (OpenCV constant)
            value: Value to set
        Returns:
            bool: True if set successfully, False otherwise
        """

        if prop_id == cv2.CAP_PROP_EXPOSURE:
            self.set_exposure(value)
            return True
        elif prop_id == cv2.CAP_PROP_BRIGHTNESS:
            return False
        elif prop_id == cv2.CAP_PROP_CONTRAST:
            return False
        elif prop_id == cv2.CAP_PROP_SATURATION:
            return False
        elif prop_id == cv2.CAP_PROP_GAIN:
            self.set_gain(value)
            return True
        elif prop_id == cv2.CAP_PROP_FPS:
            self.set_fps(value)
            return True
        elif prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            size = self.get_frame_size()
            if size:
                self.set_frame_size(int(value), size[1])
                return True
            else:
                return False
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            size = self.get_frame_size()
            if size:
                self.set_frame_size(size[0], int(value))
                return True
            else:
                return False
        else:
            return False

    def get(self, prop_id: int) -> Any:
        """
        Get a property of the capture device.
        OpenCV-compatible API method.
        
        Args:
            prop_id: Property identifier (OpenCV constant)
        Returns:
            Any: Property value or None if not available
        """
        if prop_id == cv2.CAP_PROP_EXPOSURE:
            return self.get_exposure()
        elif prop_id == cv2.CAP_PROP_BRIGHTNESS:
            # return self.get_brightness()
            return None
        elif prop_id == cv2.CAP_PROP_CONTRAST:
            # return self.get_contrast()
            return None
        elif prop_id == cv2.CAP_PROP_SATURATION:
            # return self.get_saturation()
            return None
        elif prop_id == cv2.CAP_PROP_GAIN:
            return self.get_gain()
        elif prop_id == cv2.CAP_PROP_FPS:
            return self.get_fps()
        elif prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            size = self.get_frame_size()
            if size:
                return size[0]
            else:
                return None
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            size = self.get_frame_size()
            if size:
                return size[1]
            else:
                return None
        else:
            return None

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
        """Read a frame, wrap it in a Frame with metadata, and apply processors."""
        ret, raw = self._read_implementation()

        if ret and raw is not None:
            frame: np.ndarray = Frame(
                raw,
                count=self._frame_count,
                source=str(self.source),
            )
            self._frame_count += 1
            # Use the monotonic clock (not wall-clock timestamp) so fps_actual()
            # stays correct across system clock adjustments.
            self._frame_timestamps.append(frame.monotonic)
            if hasattr(self, '_processors') and self._processors:
                for processor in self._processors:
                    frame = processor.process(frame)
        else:
            frame = raw  # type: ignore[assignment]  # raw is None when ret is False

        return ret, frame

    def fps_actual(self) -> Optional[float]:
        """Measured frame rate over the most recent frames.

        Computes the average FPS from the monotonic-clock readings
        (:attr:`Frame.monotonic`) of recently read frames (rolling window of
        up to 30). The monotonic clock is used rather than the wall-clock
        ``timestamp`` so the measurement stays accurate even if the system
        clock jumps (NTP sync, DST, manual changes). Returns ``None`` until
        at least two frames have been read.

        Returns:
            Optional[float]: Measured frames-per-second, or None if not enough
            frames have been read yet.
        """
        timestamps = self._frame_timestamps
        if len(timestamps) < 2:
            return None
        span = timestamps[-1] - timestamps[0]
        if span <= 0:
            return None
        return (len(timestamps) - 1) / span

    def wait_until_ready(self, timeout: float = 5.0) -> bool:
        """Block until the source is actually producing frames.

        Some sources (notably RTSP/network cameras) report a successful
        ``connect()`` before the underlying stream has started delivering
        frames -- the socket/session is open, but the first ``read()`` calls
        may fail for a bit while the stream negotiates. This method polls
        :meth:`read` until a frame is successfully returned or ``timeout``
        seconds have elapsed, so callers can avoid treating that startup
        window as a hard failure.

        Note:
            This consumes a frame: on success, the frame that proved
            readiness has already been read (and passed through any
            attached processors) and is not returned or buffered. Callers
            that need that particular frame should call :meth:`read` again
            immediately after, or capture it by overriding/wrapping this
            method.

        Args:
            timeout: Maximum number of seconds to wait for a frame.

        Returns:
            bool: True as soon as a frame is successfully read, False if no
            frame arrived before ``timeout`` elapsed.
        """
        deadline = time.monotonic() + timeout
        while True:
            ret, frame = self.read()
            if ret and frame is not None:
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.01)

    def __iter__(self) -> "VideoCaptureBase":
        """Return self, allowing the capture to be used as an iterator.

        Enables the idiom::

            with cap:
                for frame in cap:
                    ...  # process each frame until the source is exhausted

        Returns:
            VideoCaptureBase: self.
        """
        return self

    def __next__(self):
        """Read and return the next frame, or stop iteration.

        Returns:
            Frame: The next frame from :meth:`read`.

        Raises:
            StopIteration: If ``read()`` reports failure (``ret`` is False)
                or returns ``None`` for the frame (end of stream / no data).
        """
        ret, frame = self.read()
        if not ret or frame is None:
            raise StopIteration
        return frame


@runtime_checkable
class FrameSourceProtocol(Protocol):
    """Structural interface implemented by every FrameSource capture class.

    Use this for static type hints or ``isinstance`` checks when you want to
    accept any frame source without requiring inheritance from
    :class:`VideoCaptureBase`::

        def run(source: FrameSourceProtocol) -> None:
            source.connect()
            ok, frame = source.read()
            ...
    """

    def connect(self) -> bool: ...

    def disconnect(self) -> bool: ...

    def read(self) -> Tuple[bool, Optional[np.ndarray]]: ...

    def is_open(self) -> bool: ...

