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
import threading
import queue
import multiprocessing
from typing import Optional, Callable, Any, Tuple
import numpy as np


def simple_frame_producer(capture_source, frame_queue: queue.Queue, stop_event: threading.Event, 
                         target_fps: Optional[float] = None):
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
    
    Example:
        ```python
        from frame_source import WebcamCapture
        from frame_source.threading_utils import simple_frame_producer
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
        print("Producer: Connecting to source...")
        if not capture_source.connect():
            print("Producer: Failed to connect!")
            return
            
        print("Producer: Connected successfully")
        
        while not stop_event.is_set():
            start_time = time.time()
            
            # The key insight: just a simple, synchronous read
            success, frame = capture_source.read()
            
            if success and frame is not None:
                try:
                    # Put frame in queue (non-blocking)
                    frame_queue.put((success, frame), block=False)
                    frames_captured += 1
                except queue.Full:
                    # Drop frame if queue is full - no race condition!
                    frames_dropped += 1
            
            # Control frame rate if specified
            if frame_delay > 0:
                elapsed = time.time() - start_time
                sleep_time = max(0, frame_delay - elapsed)
                if sleep_time > 0:
                    time.sleep(sleep_time)
                
    except Exception as e:
        print(f"Producer error: {e}")
    finally:
        capture_source.disconnect()
        print(f"Producer: Captured {frames_captured} frames, dropped {frames_dropped}")


class FrameProducer:
    """
    A more sophisticated frame producer class with statistics and control.
    
    This class wraps a capture source and provides thread-safe frame production
    with built-in statistics, error handling, and flexible configuration.
    
    Example:
        ```python
        from frame_source import WebcamCapture
        from frame_source.threading_utils import FrameProducer
        
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
    
    def __init__(self, capture_source, max_queue_size: int = 10, target_fps: Optional[float] = None):
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
            'start_time': None
        }
    
    def start(self):
        """Start the frame producer thread."""
        if self._producer_thread and self._producer_thread.is_alive():
            print("Producer already running")
            return
            
        self._frame_queue = queue.Queue(maxsize=self.max_queue_size)
        self._stop_event = threading.Event()
        self._stats['start_time'] = time.time()
        
        self._producer_thread = threading.Thread(
            target=self._producer_loop,
            daemon=True
        )
        self._producer_thread.start()
        print(f"Started frame producer (queue_size={self.max_queue_size}, fps={self.target_fps})")
    
    def _producer_loop(self):
        """Internal producer loop."""
        if self._frame_queue is not None and self._stop_event is not None:
            simple_frame_producer(
                self.capture_source, 
                self._frame_queue, 
                self._stop_event, 
                self.target_fps
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
        """Get producer statistics."""
        stats = self._stats.copy()
        if stats['start_time']:
            stats['runtime'] = time.time() - stats['start_time']
            if stats['runtime'] > 0:
                stats['fps'] = stats['frames_captured'] / stats['runtime']
        return stats
    
    def _print_stats(self):
        """Print producer statistics."""
        stats = self.get_stats()
        print(f"Producer stopped. Stats: {stats}")


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
        from frame_source.threading_utils import multiprocess_frame_producer
        
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
    
    try:
        # Create capture source in the new process
        source = FrameSourceFactory.create(**source_config)
        
        if not source.connect():
            print("Multiprocess producer: Failed to connect")
            return
            
        print("Multiprocess producer: Connected")
        frames_sent = 0
        
        while not stop_event.is_set():
            success, frame = source.read()  # Simple synchronous call
            
            if success and frame is not None:
                try:
                    # Send frame to main process
                    frame_queue.put((success, frame), timeout=0.1)
                    frames_sent += 1
                except:
                    pass  # Queue full, drop frame
            
            time.sleep(0.01)  # ~100 FPS max
            
    except Exception as e:
        print(f"Multiprocess producer error: {e}")
    finally:
        if 'source' in locals():
            source.disconnect()
        print(f"Multiprocess producer finished. Frames sent: {frames_sent}")


def create_producer_consumer_pair(capture_source, consumer_function: Callable, 
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
