#!/usr/bin/env python3
"""
Example demonstrating how to use the new threading utilities from FrameSource library.
"""

import queue
import threading
import time

import cv2

from framesource.sources.webcam_capture import WebcamCapture
from framesource.threading_utils import FrameProducer, simple_frame_producer


def example_1_simple_producer():
    """Example 1: Using the simple_frame_producer function directly."""
    print("=== Example 1: Direct use of simple_frame_producer ===")

    camera = WebcamCapture(source=2)
    frame_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    # Start the producer thread using the library function
    producer_thread = threading.Thread(
        target=simple_frame_producer,
        args=(camera, frame_queue, stop_event, 30),  # 30 FPS target
        daemon=True,
    )
    producer_thread.start()

    print("Producer started. Press 'q' to quit.")

    try:
        while True:
            try:
                success, frame = frame_queue.get(timeout=0.1)
                if success and frame is not None:
                    cv2.imshow("Simple Producer Example", frame)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
            except queue.Empty:
                continue
    finally:
        stop_event.set()
        producer_thread.join(timeout=2)
        cv2.destroyAllWindows()


def example_2_frame_producer_class():
    """Example 2: Using the FrameProducer class."""
    print("\n=== Example 2: Using FrameProducer class ===")

    camera = WebcamCapture(source=2)
    producer = FrameProducer(camera, max_queue_size=10, target_fps=None)
    producer.start()

    print("FrameProducer started. Press 'q' to quit.")

    try:
        while True:
            success, frame = producer.get_frame(timeout=0.1)
            if success and frame is not None:
                cv2.imshow("FrameProducer Example", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    finally:
        producer.stop()
        cv2.destroyAllWindows()


def example_3_multiple_consumers():
    """Example 3: One producer, multiple consumers."""
    print("\n=== Example 3: Multiple consumers ===")

    camera = WebcamCapture(source=2)
    frame_queue = queue.Queue(maxsize=5)
    stop_event = threading.Event()

    # Shared state for consumers
    frame_counter = {"count": 0}
    lock = threading.Lock()

    def consumer_function():
        """Consumer that processes frames and handles display."""
        while not stop_event.is_set():
            try:
                success, frame = frame_queue.get(timeout=0.1)
                if success and frame is not None:
                    with lock:
                        frame_counter["count"] += 1
                        # Every 30 frames, show stats
                        if frame_counter["count"] % 30 == 0:
                            print(f"Processed {frame_counter['count']} frames")

                    # Simple processing - convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    cv2.imshow("Multiple Consumers Example", gray)

                    # Check for 'q' key press
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        stop_event.set()
                        break
            except queue.Empty:
                continue

    # Start producer thread
    producer_thread = threading.Thread(
        target=simple_frame_producer, args=(camera, frame_queue, stop_event, 30), daemon=True
    )

    # Start consumer thread
    consumer_thread = threading.Thread(target=consumer_function, daemon=True)

    producer_thread.start()
    consumer_thread.start()

    print("Producer-consumer pair started. Press 'q' to quit.")

    try:
        # Wait for threads to finish or user to interrupt
        while not stop_event.is_set():
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        stop_event.set()
        producer_thread.join(timeout=2)
        consumer_thread.join(timeout=2)
        cv2.destroyAllWindows()
        print(f"Total frames processed: {frame_counter['count']}")


def main():
    """Run all examples."""
    print("FrameSource Threading Utils Examples")
    print("=" * 40)

    try:
        example_1_simple_producer()
        time.sleep(1)

        example_2_frame_producer_class()
        time.sleep(1)

        example_3_multiple_consumers()

    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
