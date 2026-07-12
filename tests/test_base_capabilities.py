"""Tests for VideoCaptureBase capability additions: Frame.monotonic,
fps_actual (monotonic-based), wait_until_ready, the iterator protocol, and
the base supports_* capability flags.

Hardware-free: relies solely on the in-memory MockCapture from conftest.py.
"""

import copy
import pickle
import time

import numpy as np
import pytest

from framesource import Frame

from conftest import MockCapture


def _make_frame():
    raw = np.zeros((8, 8, 3), dtype=np.uint8)
    return Frame(raw, count=3, source="unit-test")


# ---------------------------------------------------------------------------
# Frame.monotonic
# ---------------------------------------------------------------------------

def test_frame_has_monotonic_close_to_now():
    frame = _make_frame()
    assert frame.monotonic == pytest.approx(time.monotonic(), abs=5.0)


def test_monotonic_survives_copy():
    frame = _make_frame()
    copied = frame.copy()
    assert isinstance(copied, Frame)
    assert copied.monotonic == frame.monotonic


def test_monotonic_survives_cvtcolor():
    cv2 = pytest.importorskip("cv2")
    frame = _make_frame()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if isinstance(gray, Frame):
        assert gray.monotonic == frame.monotonic


def test_monotonic_survives_pickle():
    frame = _make_frame()
    restored = pickle.loads(pickle.dumps(frame))
    assert isinstance(restored, Frame)
    assert restored.monotonic == frame.monotonic
    # timestamp should still be intact too (not clobbered by the new field).
    assert restored.timestamp == frame.timestamp


def test_monotonic_survives_deepcopy():
    frame = _make_frame()
    duplicate = copy.deepcopy(frame)
    assert isinstance(duplicate, Frame)
    assert duplicate.monotonic == frame.monotonic


def test_monotonic_explicit_value():
    raw = np.zeros((4, 4), dtype=np.uint8)
    frame = Frame(raw, monotonic=123.5)
    assert frame.monotonic == 123.5


# ---------------------------------------------------------------------------
# fps_actual
# ---------------------------------------------------------------------------

def test_fps_actual_none_before_two_reads():
    cap = MockCapture()
    cap.connect()
    assert cap.fps_actual() is None
    cap.read()
    assert cap.fps_actual() is None


def test_fps_actual_plausible_after_several_reads():
    cap = MockCapture()
    cap.connect()
    for _ in range(5):
        cap.read()
        # time.monotonic() on some platforms (e.g. Windows GetTickCount64)
        # only has ~15ms resolution; without spacing the reads out, several
        # back-to-back in-process reads can land on the same tick and yield
        # a zero span. A short sleep guarantees distinct monotonic values.
        time.sleep(0.02)
    fps = cap.fps_actual()
    assert fps is not None
    # Reads are spaced ~20ms apart, so fps should be a plausible double-digit
    # value; just sanity-check it's positive and not absurd.
    assert 0 < fps < 1000


def test_fps_actual_uses_monotonic_field():
    cap = MockCapture()
    cap.connect()
    cap._frame_timestamps.clear()
    for i in range(11):
        cap._frame_timestamps.append(i * 0.1)
    fps = cap.fps_actual()
    assert fps == pytest.approx(10.0)


# ---------------------------------------------------------------------------
# wait_until_ready
# ---------------------------------------------------------------------------

def test_wait_until_ready_true_when_connected():
    cap = MockCapture()
    cap.connect()
    assert cap.wait_until_ready(timeout=1.0) is True


def test_wait_until_ready_false_on_timeout():
    cap = MockCapture(max_frames=0)
    cap.connect()
    start = time.monotonic()
    result = cap.wait_until_ready(timeout=0.2)
    elapsed = time.monotonic() - start
    assert result is False
    # Should respect the timeout roughly (not hang, not return near-instantly
    # with a huge overshoot).
    assert elapsed < 2.0
    assert elapsed >= 0.15


# ---------------------------------------------------------------------------
# Iterator protocol
# ---------------------------------------------------------------------------

def test_iterator_yields_frames_then_stops():
    cap = MockCapture(max_frames=3)
    cap.connect()
    frames = list(cap)
    assert len(frames) == 3
    for frame in frames:
        assert isinstance(frame, Frame)


def test_iter_returns_self():
    cap = MockCapture(max_frames=1)
    cap.connect()
    assert iter(cap) is cap


# ---------------------------------------------------------------------------
# Capability flags
# ---------------------------------------------------------------------------

def test_capability_flags_base_defaults():
    cap = MockCapture()
    assert cap.supports_exposure is True
    assert cap.supports_gain is True
    assert cap.supports_depth is False
    assert cap.supports_discovery is False
    assert MockCapture.has_discovery is False
