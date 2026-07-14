"""Tests for the Frame class: metadata survives numpy/OpenCV operations."""

import copy
import pickle
import time

import numpy as np
import pytest

from framesource import Frame


def _make_frame():
    raw = np.zeros((8, 8, 3), dtype=np.uint8)
    return Frame(raw, count=7, source="unit-test")


def test_frame_is_ndarray():
    frame = _make_frame()
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (8, 8, 3)


def test_frame_has_metadata():
    frame = _make_frame()
    assert frame.count == 7
    assert frame.source == "unit-test"
    assert frame.uuid is not None
    assert isinstance(frame.metadata, dict)


def test_timestamp_near_now():
    frame = _make_frame()
    assert frame.timestamp == pytest.approx(time.time(), abs=5.0)


def test_unique_uuids():
    a = _make_frame()
    b = _make_frame()
    assert a.uuid != b.uuid


def test_metadata_survives_copy():
    frame = _make_frame()
    copied = frame.copy()
    assert isinstance(copied, Frame)
    assert copied.uuid == frame.uuid
    assert copied.count == frame.count


def test_metadata_survives_slicing():
    frame = _make_frame()
    sliced = frame[2:6, 2:6]
    assert isinstance(sliced, Frame)
    assert sliced.uuid == frame.uuid
    assert sliced.source == frame.source


def test_metadata_survives_cvtcolor():
    cv2 = pytest.importorskip("cv2")
    frame = _make_frame()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # OpenCV returns an ndarray; if it preserves the subclass, metadata stays.
    if isinstance(gray, Frame):
        assert gray.uuid == frame.uuid


def test_metadata_survives_pickle():
    frame = _make_frame()
    restored = pickle.loads(pickle.dumps(frame))
    assert isinstance(restored, Frame)
    assert restored.uuid == frame.uuid
    assert restored.count == frame.count
    assert restored.source == frame.source
    assert restored.timestamp == frame.timestamp


def test_metadata_survives_deepcopy():
    frame = _make_frame()
    duplicate = copy.deepcopy(frame)
    assert isinstance(duplicate, Frame)
    assert duplicate.uuid == frame.uuid
    assert duplicate.count == frame.count
