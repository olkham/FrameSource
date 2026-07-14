"""External-threading tests for FrameSource.

These verify the recommended concurrency model: capture classes are synchronous
and concurrency lives in the caller via `simple_frame_producer` and a queue.
No hardware is required — `MockCapture` provides deterministic frames.
"""

import queue
import threading
import time

import pytest
from conftest import MockCapture

from framesource import Frame, FrameProducer
from framesource.threading_utils import simple_frame_producer


def test_simple_frame_producer_fills_queue():
    camera = MockCapture()
    frame_queue: queue.Queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()

    producer = threading.Thread(
        target=simple_frame_producer,
        args=(camera, frame_queue, stop_event),
        daemon=True,
    )
    producer.start()

    received = 0
    deadline = time.time() + 2.0
    while received < 5 and time.time() < deadline:
        try:
            success, frame = frame_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if success and frame is not None:
            assert isinstance(frame, Frame)
            received += 1

    stop_event.set()
    producer.join(timeout=2)
    assert received >= 5


def test_frame_producer_stats_include_avg_latency():
    camera = MockCapture()
    producer = FrameProducer(camera, max_queue_size=10, target_fps=None)
    producer.start()

    # Drain frames so the producer keeps capturing.
    deadline = time.time() + 2.0
    drained = 0
    while drained < 5 and time.time() < deadline:
        ok, _ = producer.get_frame(timeout=0.1)
        if ok:
            drained += 1

    producer.stop()
    stats = producer.get_stats()
    assert "avg_latency" in stats
    assert stats["frames_captured"] >= 1
    assert "fps" in stats


def test_frame_producer_reports_drops_on_small_queue():
    # A tiny queue that is never drained should accumulate drops.
    camera = MockCapture()
    producer = FrameProducer(camera, max_queue_size=1, target_fps=None)
    producer.start()
    time.sleep(0.5)
    producer.stop()

    stats = producer.get_stats()
    assert stats["frames_dropped"] > 0


@pytest.mark.hardware
def test_webcam_external_threading():
    """Opt-in hardware test (run with `pytest -m hardware`)."""
    from framesource import WebcamCapture

    camera = WebcamCapture(source=0)
    frame_queue: queue.Queue = queue.Queue(maxsize=10)
    stop_event = threading.Event()
    producer = threading.Thread(
        target=simple_frame_producer,
        args=(camera, frame_queue, stop_event, 30),
        daemon=True,
    )
    producer.start()
    try:
        success, frame = frame_queue.get(timeout=5.0)
        assert success and frame is not None
    finally:
        stop_event.set()
        producer.join(timeout=2)
