"""Tests for the DeviceInfo dataclass (Step 7.3) and its dict-compatibility."""

import pytest

from framesource.discovery import DeviceInfo


def _webcam_like():
    return DeviceInfo(
        device_id="700:0:/dev/video0",
        index=0,
        name="Integrated Camera",
        driver="opencv",
        id_stable=False,
        metadata={"backend_index": 700, "backend_name": "DSHOW"},
    )


def test_getitem_field_and_alias():
    d = _webcam_like()
    assert d["id"] == "700:0:/dev/video0"          # alias -> device_id
    assert d["device_id"] == "700:0:/dev/video0"
    assert d["name"] == "Integrated Camera"
    assert d["index"] == 0
    assert d["driver"] == "opencv"


def test_getitem_metadata_fallback():
    d = _webcam_like()
    assert d["backend_name"] == "DSHOW"
    assert d["backend_index"] == 700


def test_getitem_missing_raises_keyerror():
    d = _webcam_like()
    with pytest.raises(KeyError):
        _ = d["does_not_exist"]


def test_get_with_default():
    d = _webcam_like()
    assert d.get("name") == "Integrated Camera"
    assert d.get("backend_name") == "DSHOW"
    assert d.get("missing") is None
    assert d.get("missing", "fallback") == "fallback"


def test_contains():
    d = _webcam_like()
    assert "id" in d
    assert "device_id" in d
    assert "name" in d
    assert "backend_name" in d
    assert "nope" not in d


def test_keys_and_items():
    d = _webcam_like()
    keys = set(d.keys())
    assert {"id", "index", "name", "backend_name"} <= keys
    items = dict(d.items())
    assert items["id"] == "700:0:/dev/video0"
    assert items["backend_name"] == "DSHOW"


def test_as_dict_roundtrips():
    d = _webcam_like()
    dumped = d.as_dict()
    assert dumped["device_id"] == "700:0:/dev/video0"
    assert dumped["metadata"] == {"backend_index": 700, "backend_name": "DSHOW"}
    restored = DeviceInfo(**dumped)
    assert restored == d


def test_defaults():
    d = DeviceInfo(device_id="x")
    assert d.index is None
    assert d.name == ""
    assert d.driver == ""
    assert d.id_stable is False
    assert d.metadata == {}


def test_webcam_discover_returns_list_of_deviceinfo():
    """webcam discovery is machine-safe (returns [] when the enumerator or
    cameras are unavailable); whatever it returns must be a list of
    dict-compatible DeviceInfo instances."""
    from framesource.sources.webcam_capture import WebcamCapture

    result = WebcamCapture.discover()
    assert isinstance(result, list)
    for dev in result:
        assert isinstance(dev, DeviceInfo)
        # Legacy dict-style access must keep working.
        assert dev["id"] is not None
        _ = dev["name"]
        _ = dev["index"]
