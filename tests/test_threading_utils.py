"""Tests for the async and fan-out threading helpers.

Hardware-free: `MockCapture` supplies deterministic frames. Async code is
driven with `asyncio.run(...)` inside plain test functions (no pytest-asyncio).
"""

import asyncio
import queue
import time

from framesource.threading_utils import AsyncFrameSource, SharedProducer
from framesource.sources.video_capture_base import FrameSourceProtocol

from conftest import MockCapture


def test_async_frame_source_context_manager_reads_frames():
    camera = MockCapture(max_frames=10)

    async def run():
        received = []
        async with AsyncFrameSource(camera) as src:
            for _ in range(5):
                ret, frame = await src.read()
                if ret:
                    received.append(frame)
        return received

    received = asyncio.run(run())
    assert len(received) == 5
    assert all(f is not None for f in received)
    # After the `async with` block the source is disconnected and closed.
    assert camera.is_connected is False


def test_async_frame_source_lazy_executor():
    # Merely constructing the adapter must not create a worker thread.
    src = AsyncFrameSource(MockCapture())
    assert src._executor is None
    # close() is safe even though the executor was never created.
    src.close()
    assert src._executor is None


def test_async_frame_source_close_shuts_down_executor():
    camera = MockCapture(max_frames=3)
    src = AsyncFrameSource(camera)

    async def run():
        await src.connect()
        await src.read()

    asyncio.run(run())
    assert src._executor is not None
    src.close()
    assert src._executor is None


def test_shared_producer_fans_out_to_two_subscribers():
    camera = MockCapture()
    shared = SharedProducer(camera)

    q1 = shared.subscribe(maxsize=10)
    q2 = shared.subscribe(maxsize=10)

    shared.start()
    try:
        got1 = _drain(q1, count=3, deadline=2.0)
        got2 = _drain(q2, count=3, deadline=2.0)
    finally:
        shared.stop()

    assert got1 >= 3
    assert got2 >= 3
    # start() connected the (initially unconnected) source; stop() leaves it
    # to the caller, so it is still connected here.
    assert camera.is_connected is True
    camera.disconnect()


def test_shared_producer_unsubscribe_stops_delivery():
    camera = MockCapture()
    shared = SharedProducer(camera)
    q = shared.subscribe(maxsize=10)

    shared.start()
    try:
        assert _drain(q, count=1, deadline=2.0) >= 1
        assert shared.unsubscribe(q) is True
        # Drain whatever is buffered, then confirm nothing new arrives.
        _drain_all(q)
        time.sleep(0.2)
        assert q.qsize() == 0
        # Unsubscribing an unknown queue returns False.
        assert shared.unsubscribe(queue.Queue()) is False
    finally:
        shared.stop()
        camera.disconnect()


def test_shared_producer_stats_keys_present():
    camera = MockCapture()
    shared = SharedProducer(camera)
    shared.subscribe(maxsize=1)  # tiny queue, never drained -> drops accrue
    shared.start()
    time.sleep(0.3)
    shared.stop()
    camera.disconnect()

    stats = shared.get_stats()
    for key in ('frames_captured', 'frames_dropped', 'drops_per_subscriber',
                'runtime', 'fps'):
        assert key in stats
    assert stats['frames_captured'] >= 1


def test_shared_producer_stop_joins_cleanly():
    camera = MockCapture()
    shared = SharedProducer(camera)
    shared.subscribe(maxsize=10)
    shared.start()
    time.sleep(0.1)
    shared.stop()
    camera.disconnect()
    assert shared._thread is not None
    assert shared._thread.is_alive() is False


def test_mock_capture_satisfies_protocol():
    assert isinstance(MockCapture(), FrameSourceProtocol)


def _drain(q: "queue.Queue", count: int, deadline: float) -> int:
    """Pull up to `count` successful frames from `q` within `deadline` seconds."""
    received = 0
    end = time.time() + deadline
    while received < count and time.time() < end:
        try:
            ret, frame = q.get(timeout=0.1)
        except queue.Empty:
            continue
        if ret and frame is not None:
            received += 1
    return received


def _drain_all(q: "queue.Queue") -> None:
    """Empty a queue without blocking."""
    while True:
        try:
            q.get_nowait()
        except queue.Empty:
            return
