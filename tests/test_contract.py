"""Tests for the FrameSource capture contract using the hardware-free MockCapture."""

import numpy as np
from conftest import MockCapture

from framesource import Frame, FrameSourceProtocol
from framesource.sources.video_capture_base import VideoCaptureBase


def test_connect_sets_open_state():
    cap = MockCapture()
    assert cap.is_open() is False
    assert cap.connect() is True
    assert cap.is_open() is True
    assert cap.isOpened() is True


def test_disconnect_clears_open_state():
    cap = MockCapture()
    cap.connect()
    assert cap.disconnect() is True
    assert cap.is_open() is False


def test_read_returns_frame_when_connected():
    cap = MockCapture()
    cap.connect()
    ok, frame = cap.read()
    assert ok is True
    assert isinstance(frame, Frame)
    assert frame.source == "mock"


def test_read_fails_when_not_connected():
    cap = MockCapture()
    ok, frame = cap.read()
    assert ok is False
    assert frame is None


def test_count_increments_monotonically():
    cap = MockCapture()
    cap.connect()
    counts = []
    for _ in range(5):
        _, frame = cap.read()
        counts.append(frame.count)
    assert counts == [0, 1, 2, 3, 4]


def test_context_manager_connects_and_disconnects():
    cap = MockCapture()
    with cap as c:
        assert c.is_open() is True
        ok, frame = c.read()
        assert ok is True
    assert cap.is_open() is False


def test_release_disconnects():
    cap = MockCapture()
    cap.connect()
    cap.release()
    assert cap.is_open() is False


def test_satisfies_protocol():
    assert isinstance(MockCapture(), FrameSourceProtocol)
    assert issubclass(MockCapture, VideoCaptureBase)


def test_fps_actual_none_before_two_reads():
    cap = MockCapture()
    cap.connect()
    assert cap.fps_actual() is None
    cap.read()
    assert cap.fps_actual() is None


def test_fps_actual_computes_from_timestamps():
    cap = MockCapture()
    cap.connect()
    # Inject deterministic timestamps spaced 0.1s apart (10 FPS).
    cap._frame_timestamps.clear()
    for i in range(11):
        cap._frame_timestamps.append(i * 0.1)
    assert cap.fps_actual() == np.float64(10.0) or abs(cap.fps_actual() - 10.0) < 1e-6
