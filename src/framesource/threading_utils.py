"""
Threading utilities for FrameSource library.

This module provides helper functions and classes for implementing
external threading patterns with FrameSource capture sources.

These utilities demonstrate best practices for:
- Producer-consumer patterns
- Thread-safe frame handling
- Queue-based communication
- Race condition avoidance
"""

import time
import asyncio
import threading
import queue
import multiprocessing
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Callable, Any, Tuple, List, Dict
import numpy as np

from .sources.video_capture_base import FrameSourceProtocol

logger = logging.getLogger(__name__)

# Drop rate above this fraction triggers a warning (queue can't keep up).
_HIGH_DROP_RATE_THRESHOLD = 0.1


def simple_frame_producer(capture_source: FrameSourceProtocol, frame_queue: queue.Queue,
                         stop_event: threading.Event,
                         target_fps: Optional[float] = None, stats: Optional[dict] = None):
    """
    Simple producer function that runs in a thread.
    
    This demonstrates the recommended approach for external threading with FrameSource.
    The producer connects to a capture source, reads frames synchronously, and puts
    them in a thread-safe queue for consumers.
    
    Args:
        capture_source: Any FrameSource capture object (WebcamCapture, RealsenseCapture, etc.)
        frame_queue: Thread-safe queue to put frames into
        stop_event: Threading event to signal when to stop
        target_fps: Optional target frame rate (None for unlimited)
        stats: Optional dict updated in-place with running statistics
            (frames_captured, frames_dropped, _latency_sum, _latency_count)
    
    Example:
        ```python
        from framesource import WebcamCapture
        from framesource.threading_utils import simple_frame_producer
        import queue
        import threading
        
        camera = WebcamCapture(source=0)
        frame_queue = queue.Queue(maxsize=10)
        stop_event = threading.Event()
        
        producer_thread = threading.Thread(
            target=simple_frame_producer,
            args=(camera, frame_queue, stop_event, 30),  # 30 FPS
            daemon=True
        )
        producer_thread.start()
        
        # Consumer loop
        while True:
            success, frame = frame_queue.get()
            if success:
                process_frame(frame)
        ```
    """
    frame_delay = 1.0 / target_fps if target_fps else 0.0
    frames_captured = 0
    frames_dropped = 0
    
    try:
        logger.info("Producer: connecting to source...")
        if not capture_source.connect():
            logger.error("Producer: failed to connect to source")
            return
            
        logger.info("Producer: connected successfully")
        
        while not stop_event.is_set():
            start_time = time.time()
            
            # The key insight: just a simple, synchronous read
            success, frame = capture_source.read()
            
            if success and frame is not None:
                try:
                    # Put frame in queue (non-blocking)
                    frame_queue.put((success, frame), block=False)
                    frames_captured += 1
                    if stats is not None:
                        stats['frames_captured'] = frames_captured
                        # Latency from when the frame was stamped to now.
                        timestamp = getattr(frame, 'timestamp', None)
                        if timestamp is not None:
                            stats['_latency_sum'] = stats.get('_latency_sum', 0.0) + (time.time() - timestamp)
                            stats['_latency_count'] = stats.get('_latency_count', 0) + 1
                except queue.Full:
                    # Drop frame if queue is full - no race condition!
                    frames_dropped += 1
                    if stats is not None:
                        stats['frames_dropped'] = frames_dropped
            
            # Control frame rate if specified
            if frame_delay > 0:
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_delay - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
    except Exception as e:
        logger.exception("Producer error: %s", e)
        if stats is not None:
            stats['errors'] = stats.get('errors', 0) + 1
    finally:
        capture_source.disconnect()
        total = frames_captured + frames_dropped
        if total > 0 and (frames_dropped / total) > _HIGH_DROP_RATE_THRESHOLD:
            logger.warning(
                "Producer: high drop rate %.1f%% (%d/%d) - consumer cannot keep up; "
                "increase queue size or processing speed.",
                100.0 * frames_dropped / total, frames_dropped, total,
            )
        logger.info("Producer: captured %d frames, dropped %d", frames_captured, frames_dropped)


class FrameProducer:
    """
    A more sophisticated frame producer class with statistics and control.
    
    This class wraps a capture source and provides thread-safe frame production
    with built-in statistics, error handling, and flexible configuration.
    
    Example:
        ```python
        from framesource import WebcamCapture
        from framesource.threading_utils import FrameProducer
        
        camera = WebcamCapture(source=0)
        producer = FrameProducer(camera, max_queue_size=10, target_fps=30)
        
        producer.start()
        
        # Get frames
        while True:
            success, frame = producer.get_frame(timeout=0.1)
            if success:
                process_frame(frame)
        
        producer.stop()
        ```
    """
    
    def __init__(self, capture_source: FrameSourceProtocol, max_queue_size: int = 10,
                 target_fps: Optional[float] = None):
        """
        Initialize the frame producer.
        
        Args:
            capture_source: FrameSource capture object
            max_queue_size: Maximum size of the frame queue
            target_fps: Target frame rate (None for unlimited)
        """
        self.capture_source = capture_source
        self.max_queue_size = max_queue_size
        self.target_fps = target_fps
        
        self._frame_queue = None
        self._stop_event = None
        self._producer_thread = None
        
        self._stats = {
            'frames_captured': 0,
            'frames_dropped': 0,
            'errors': 0,
            'start_time': None,
            '_latency_sum': 0.0,
            '_latency_count': 0,
        }
    
    def start(self):
        """Start the frame producer thread."""
        if self._producer_thread and self._producer_thread.is_alive():
            logger.warning("Producer already running")
            return
            
        self._frame_queue = queue.Queue(maxsize=self.max_queue_size)
        self._stop_event = threading.Event()
        self._stats['start_time'] = time.time()
        
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            daemon=True
        )
        self._producer_thread.start()
        logger.info("Started frame producer (queue_size=%s, fps=%s)", self.max_queue_size, self.target_fps)
    
    def _producer_loop(self):
        """Internal producer loop."""
        if self._frame_queue is not None and self._stop_event is not None:
            simple_frame_producer(
                self.capture_source, 
                self._frame_queue, 
                self._stop_event, 
                self.target_fps,
                self._stats,
            )
    
    def get_frame(self, timeout: float = 0.1) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Get the next frame from the producer queue.
        
        Args:
            timeout: Maximum time to wait for a frame
            
        Returns:
            Tuple[bool, Optional[np.ndarray]]: (success, frame)
        """
        if not self._frame_queue:
            return False, None
            
        try:
            return self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return False, None
    
    def stop(self):
        """Stop the producer thread."""
        if self._stop_event:
            self._stop_event.set()
        if self._producer_thread:
            self._producer_thread.join(timeout=2)
        
        self._print_stats()
    
    def get_stats(self) -> dict:
        """Get producer statistics.

        Returns a dict with ``frames_captured``, ``frames_dropped``, ``errors``,
        ``runtime``, ``fps`` and ``avg_latency`` (mean seconds between a frame
        being stamped and enqueued). Internal accumulator keys (prefixed with
        ``_``) are omitted.
        """
        stats = self._stats.copy()
        latency_sum = stats.pop('_latency_sum', 0.0)
        latency_count = stats.pop('_latency_count', 0)
        stats['avg_latency'] = (latency_sum / latency_count) if latency_count else None
        if stats['start_time']:
            stats['runtime'] = time.time() - stats['start_time']
            if stats['runtime'] > 0:
                stats['fps'] = stats['frames_captured'] / stats['runtime']
        return stats
    
    def _print_stats(self):
        """Log producer statistics."""
        stats = self.get_stats()
        logger.info("Producer stopped. Stats: %s", stats)


def multiprocess_frame_producer(source_config: dict, frame_queue, stop_event):
    """
    Producer function for multiprocessing (bypasses GIL).
    
    This function runs in a separate process and communicates via
    multiprocessing queues and events.
    
    Args:
        source_config: Dictionary with capture source configuration
        frame_queue: multiprocessing.Queue for frames
        stop_event: multiprocessing.Event to signal stop
    
    Example:
        ```python
        import multiprocessing
        from framesource.threading_utils import multiprocess_frame_producer
        
        source_config = {'source': 0, 'width': 640, 'height': 480}
        frame_queue = multiprocessing.Queue(maxsize=10)
        stop_event = multiprocessing.Event()
        
        producer_process = multiprocessing.Process(
            target=multiprocess_frame_producer,
            args=(source_config, frame_queue, stop_event)
        )
        producer_process.start()
        
        # Consumer in main process
        while True:
            success, frame = frame_queue.get()
            if success:
                process_frame(frame)
        ```
    """
    # Import here to avoid issues with multiprocessing
    from .factory import FrameSourceFactory

    source = None
    frames_sent = 0

    try:
        # Create capture source in the new process
        source = FrameSourceFactory.create(**source_config)

        if not source.connect():
            logger.error("Multiprocess producer: failed to connect")
            return

        logger.info("Multiprocess producer: connected")

        while not stop_event.is_set():
            success, frame = source.read()  # Simple synchronous call

            if success and frame is not None:
                try:
                    # Send frame to main process
                    frame_queue.put((success, frame), timeout=0.1)
                    frames_sent += 1
                except queue.Full:
                    pass  # Queue full, drop frame

            time.sleep(0.01)  # ~100 FPS max

    except Exception as e:
        logger.exception("Multiprocess producer error: %s", e)
    finally:
        if source is not None:
            source.disconnect()
        logger.info("Multiprocess producer finished. Frames sent: %s", frames_sent)


def create_producer_consumer_pair(capture_source: FrameSourceProtocol, consumer_function: Callable,
                                 max_queue_size: int = 10, target_fps: Optional[float] = None):
    """
    Convenience function to create a producer-consumer pair.
    
    Args:
        capture_source: FrameSource capture object
        consumer_function: Function that processes frames (receives success, frame)
        max_queue_size: Maximum queue size
        target_fps: Target frame rate
    
    Returns:
        Tuple of (producer_thread, stop_event) for control
    
    Example:
        ```python
        def my_processor(success, frame):
            if success:
                cv2.imshow("Frame", frame)
                cv2.waitKey(1)
        
        camera = WebcamCapture(source=0)
        producer_thread, stop_event = create_producer_consumer_pair(
            camera, my_processor, max_queue_size=5, target_fps=30
        )
        
        # Let it run for a while
        time.sleep(10)
        
        # Stop it
        stop_event.set()
        producer_thread.join()
        ```
    """
    frame_queue = queue.Queue(maxsize=max_queue_size)
    stop_event = threading.Event()
    
    def consumer_loop():
        while not stop_event.is_set():
            try:
                success, frame = frame_queue.get(timeout=0.1)
                consumer_function(success, frame)
            except queue.Empty:
                continue
    
    # Start producer
    producer_thread = threading.Thread(
        target=simple_frame_producer,
        args=(capture_source, frame_queue, stop_event, target_fps),
        daemon=True
    )
    
    # Start consumer
    consumer_thread = threading.Thread(target=consumer_loop, daemon=True)
    
    producer_thread.start()
    consumer_thread.start()

    return producer_thread, stop_event


class AsyncFrameSource:
    """Opt-in asyncio adapter around a synchronous FrameSource capture object.

    FrameSource capture objects are synchronous by design ("bring-your-own-
    concurrency"): the core never spawns threads on your behalf. This adapter
    is a thin, explicit bridge for callers whose application is built on
    ``asyncio``. It offloads the blocking ``connect``/``read``/``disconnect``
    calls onto a dedicated single-worker thread so they do not stall the event
    loop.

    The executor is created lazily on first use, so sync-only users who merely
    construct (but never ``await``) an :class:`AsyncFrameSource` never pay for a
    thread. A single worker is used deliberately: capture objects are not
    thread-safe, so all calls are serialized onto the same thread.

    Note:
        This does not make the capture source itself asynchronous — it only
        moves the synchronous work off the event loop thread. The core remains
        synchronous; async support is entirely opt-in and lives here.

    Example:
        ```python
        import asyncio
        from framesource import WebcamCapture
        from framesource.threading_utils import AsyncFrameSource

        async def main():
            async with AsyncFrameSource(WebcamCapture(source=0)) as src:
                for _ in range(100):
                    ret, frame = await src.read()
                    if ret:
                        ...  # process frame without blocking the loop

        asyncio.run(main())
        ```
    """

    def __init__(self, capture_source: FrameSourceProtocol):
        """Initialize the adapter.

        Args:
            capture_source: Any synchronous FrameSource capture object.
        """
        self.capture_source = capture_source
        self._executor: Optional[ThreadPoolExecutor] = None

    def _ensure_executor(self) -> ThreadPoolExecutor:
        """Lazily create the single-worker executor on first use."""
        if self._executor is None:
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix='framesource-async'
            )
        return self._executor

    async def connect(self) -> bool:
        """Connect to the underlying source without blocking the event loop.

        Returns:
            bool: True if the connection succeeded.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._ensure_executor(), self.capture_source.connect
        )

    async def read(self):
        """Read one frame without blocking the event loop.

        Returns:
            Tuple[bool, Optional[Frame]]: The OpenCV-style ``(ret, frame)``
            tuple returned by the underlying source.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._ensure_executor(), self.capture_source.read
        )

    async def disconnect(self) -> bool:
        """Disconnect from the underlying source without blocking the loop.

        Returns:
            bool: True if the disconnection succeeded.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._ensure_executor(), self.capture_source.disconnect
        )

    def close(self) -> None:
        """Shut down the worker thread, if one was created.

        Safe to call even if the executor was never created (sync-only use).
        """
        if self._executor is not None:
            self._executor.shutdown(wait=False, cancel_futures=True)
            self._executor = None

    async def __aenter__(self) -> "AsyncFrameSource":
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            await self.disconnect()
        finally:
            self.close()


class SharedProducer:
    """Explicit one-producer / many-consumer fan-out for a single source.

    A single daemon thread reads frames from one capture source and fans each
    ``(ret, frame)`` tuple out to every subscribed queue. This lets several
    independent consumers (e.g. a live UI preview and a background recorder)
    share one camera without each opening its own connection and without any
    hidden global state or reference-counting magic — you explicitly
    :meth:`subscribe`, :meth:`start`, :meth:`stop`, and :meth:`unsubscribe`.

    Each subscriber gets its own bounded queue. Frames are delivered with
    ``put(block=False)``; if a subscriber's queue is full its frame is dropped
    (counted per-queue) so a slow consumer never stalls the producer or the
    other consumers.

    Lifecycle:
        :meth:`start` connects the source if it is not already open (checked via
        ``is_open()``). :meth:`stop` does **not** disconnect — the caller owns
        the source's lifecycle and should disconnect it when finished. This
        keeps ownership explicit when the same source is shared or reused.

    Example:
        ```python
        from framesource import WebcamCapture
        from framesource.threading_utils import SharedProducer

        camera = WebcamCapture(source=0)
        shared = SharedProducer(camera, target_fps=30)

        ui_q = shared.subscribe(maxsize=2)        # UI: latest frames only
        rec_q = shared.subscribe(maxsize=120)     # recorder: bigger buffer

        shared.start()
        try:
            ok, frame = ui_q.get(timeout=1.0)     # feed the preview widget
            ok, frame = rec_q.get(timeout=1.0)    # write to the video file
        finally:
            shared.stop()
            camera.disconnect()                   # caller owns the source
        ```
    """

    def __init__(self, capture_source: FrameSourceProtocol,
                 target_fps: Optional[float] = None):
        """Initialize the shared producer.

        Args:
            capture_source: FrameSource capture object to read from.
            target_fps: Optional target frame rate for pacing (None = unlimited).
        """
        self.capture_source = capture_source
        self.target_fps = target_fps

        self._subscribers: List[queue.Queue] = []
        self._drops: Dict[int, int] = {}
        self._lock = threading.Lock()

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frames_captured = 0
        self._start_time: Optional[float] = None

    def subscribe(self, maxsize: int = 10) -> queue.Queue:
        """Register a new consumer queue and return it.

        May be called before or after :meth:`start`.

        Args:
            maxsize: Maximum size of the created queue (0 = unbounded).

        Returns:
            queue.Queue: A newly created queue that will receive
            ``(ret, frame)`` tuples.
        """
        q: queue.Queue = queue.Queue(maxsize=maxsize)
        with self._lock:
            self._subscribers.append(q)
            self._drops[id(q)] = 0
        return q

    def unsubscribe(self, q: queue.Queue) -> bool:
        """Deregister a consumer queue so it stops receiving frames.

        Args:
            q: A queue previously returned by :meth:`subscribe`.

        Returns:
            bool: True if the queue was registered and has been removed.
        """
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)
                self._drops.pop(id(q), None)
                return True
        return False

    def start(self) -> None:
        """Start the producer thread (connecting the source if needed)."""
        if self._thread and self._thread.is_alive():
            logger.warning("SharedProducer already running")
            return

        if not self.capture_source.is_open():
            if not self.capture_source.connect():
                logger.error("SharedProducer: failed to connect to source")
                return

        self._stop_event = threading.Event()
        self._start_time = time.time()
        self._thread = threading.Thread(target=self._producer_loop, daemon=True)
        self._thread.start()
        logger.info("Started SharedProducer (fps=%s)", self.target_fps)

    def _producer_loop(self) -> None:
        """Internal producer loop: read then fan out to every subscriber."""
        frame_delay = 1.0 / self.target_fps if self.target_fps else 0.0
        try:
            while not self._stop_event.is_set():
                start_time = time.time()

                ret, frame = self.capture_source.read()
                if ret and frame is not None:
                    self._frames_captured += 1
                    with self._lock:
                        for q in self._subscribers:
                            try:
                                q.put((ret, frame), block=False)
                            except queue.Full:
                                self._drops[id(q)] = self._drops.get(id(q), 0) + 1

                if frame_delay > 0:
                    elapsed = time.time() - start_time
                    sleep_time = frame_delay - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)
        except Exception as e:
            logger.exception("SharedProducer error: %s", e)

    def stop(self) -> None:
        """Signal the producer thread to stop and join it (2s timeout).

        Does not disconnect the source; the caller owns its lifecycle.
        """
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=2)

    def get_stats(self) -> dict:
        """Return producer statistics.

        Returns:
            dict: With keys ``frames_captured``, ``frames_dropped`` (total
            across all subscribers), ``drops_per_subscriber`` (list of per-queue
            drop counts), ``runtime`` (seconds) and ``fps`` (captured frames per
            second).
        """
        with self._lock:
            per_subscriber = list(self._drops.values())
        total_drops = sum(per_subscriber)
        runtime = (time.time() - self._start_time) if self._start_time else 0.0
        fps = (self._frames_captured / runtime) if runtime > 0 else 0.0
        return {
            'frames_captured': self._frames_captured,
            'frames_dropped': total_drops,
            'drops_per_subscriber': per_subscriber,
            'runtime': runtime,
            'fps': fps,
        }
