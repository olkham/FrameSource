"""FrameSourceFactory behaviour tests using the hardware-free MockCapture."""

import pytest

from framesource import FrameSourceFactory

from conftest import MockCapture

MOCK_TYPE = "mock"


@pytest.fixture(autouse=True)
def register_mock():
    FrameSourceFactory.register_capture_type(MOCK_TYPE, MockCapture)
    yield
    FrameSourceFactory.unregister_capture_type(MOCK_TYPE)


def test_create_connects_by_default():
    cap = FrameSourceFactory.create(MOCK_TYPE, source_id="mock")
    assert cap.is_connected


def test_create_connect_false_skips_connect_and_does_not_leak_kwarg():
    cap = FrameSourceFactory.create(MOCK_TYPE, source_id="mock", connect=False)
    assert not cap.is_connected
    assert "connect" not in cap.config


def test_create_unknown_type_raises_with_available_types():
    with pytest.raises(ValueError, match=MOCK_TYPE):
        FrameSourceFactory.create("definitely_not_registered", source_id=0)


def test_deprecated_source_param_warns_and_works():
    with pytest.warns(DeprecationWarning, match="'source' is deprecated"):
        cap = FrameSourceFactory.create(MOCK_TYPE, source="mock", connect=False)
    assert cap.source == "mock"


def test_deprecated_capture_type_param_warns_and_works():
    with pytest.warns(DeprecationWarning, match="'capture_type' is deprecated"):
        cap = FrameSourceFactory.create(capture_type=MOCK_TYPE, source_id="mock", connect=False)
    assert isinstance(cap, MockCapture)


def test_kwarg_capture_type_and_source_still_work():
    with pytest.warns(DeprecationWarning):
        cap = FrameSourceFactory.create(**{"capture_type": MOCK_TYPE, "source": "mock",
                                           "connect": False})
    assert isinstance(cap, MockCapture)
    assert cap.source == "mock"


def test_register_rejects_non_capture_class():
    with pytest.raises(ValueError):
        FrameSourceFactory.register_capture_type("bad", dict)


def test_unregister_unknown_raises():
    with pytest.raises(ValueError):
        FrameSourceFactory.unregister_capture_type("never_registered")
