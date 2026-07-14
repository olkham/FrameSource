"""Tests for the framesource.errors exception hierarchy (hardware-free)."""

import builtins

import pytest

from framesource import FrameSourceFactory
from framesource.errors import (
    FrameSourceError,
    MissingDependencyError,
    NotConnectedError,
    UnknownSourceTypeError,
)


def test_missing_dependency_error_is_import_error():
    assert issubclass(MissingDependencyError, ImportError)
    assert issubclass(MissingDependencyError, FrameSourceError)


def test_unknown_source_type_error_is_value_error():
    assert issubclass(UnknownSourceTypeError, ValueError)
    assert issubclass(UnknownSourceTypeError, FrameSourceError)


def test_not_connected_error_is_frame_source_error():
    assert issubclass(NotConnectedError, FrameSourceError)


def test_all_errors_derive_from_frame_source_error():
    for exc_type in (MissingDependencyError, UnknownSourceTypeError, NotConnectedError):
        assert issubclass(exc_type, FrameSourceError)
    assert issubclass(FrameSourceError, RuntimeError)


def test_missing_dependency_error_message_contains_package_and_install_hint():
    err = MissingDependencyError("pyaudio", extra="audio")
    message = str(err)
    assert "pyaudio" in message
    assert "pip install framesource[audio]" in message


def test_missing_dependency_error_without_extra_omits_hint():
    err = MissingDependencyError("somepkg")
    message = str(err)
    assert "somepkg" in message
    assert "pip install" not in message


def test_missing_dependency_error_includes_details():
    err = MissingDependencyError(
        "librosa/soundfile/pyaudio", extra="audio", details="No module named librosa"
    )
    message = str(err)
    assert "No module named librosa" in message


def test_missing_dependency_error_stores_attributes():
    err = MissingDependencyError("pyaudio", extra="audio")
    assert err.package == "pyaudio"
    assert err.extra == "audio"


def test_factory_unknown_type_raises_unknown_source_type_error():
    with pytest.raises(UnknownSourceTypeError):
        FrameSourceFactory.create("definitely_not_a_type", source_id=0)


def test_factory_unknown_type_still_caught_by_value_error():
    # Backwards compatibility: old user code catching plain ValueError
    # must keep working.
    with pytest.raises(ValueError):
        FrameSourceFactory.create("definitely_not_a_type", source_id=0)


# --------------------------------------------------------------------------- #
# Vendor sources: MissingDependencyError wiring (Step C1)
#
# Each of these simulates the vendor SDK being absent (regardless of whether
# it is actually installed in the environment running the tests) by
# monkeypatching ``builtins.__import__`` to raise ImportError only for the
# targeted module name, mirroring the existing PyYAML-absence test in
# test_from_config.py. This keeps the tests deterministic and hardware-free.
# --------------------------------------------------------------------------- #


def _poison_import(monkeypatch, *blocked_names):
    """Make ``import <name>`` raise ImportError for any of ``blocked_names``.

    Matches exact names and dotted submodule imports (e.g. blocking
    'harvesters' also blocks 'harvesters.core').
    """
    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if any(name == blocked or name.startswith(blocked + ".") for blocked in blocked_names):
            raise ImportError(f"No module named '{name}'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)


def test_basler_missing_pypylon_raises_missing_dependency_error(monkeypatch):
    _poison_import(monkeypatch, "pypylon")
    from framesource.sources.basler_capture import BaslerCapture

    with pytest.raises(MissingDependencyError) as excinfo:
        BaslerCapture()

    err = excinfo.value
    assert isinstance(err, ImportError)
    message = str(err)
    assert "pypylon" in message
    assert "pip install framesource[basler]" in message


def test_ximea_missing_sdk_raises_missing_dependency_error(monkeypatch):
    _poison_import(monkeypatch, "ximea")
    from framesource.sources.ximea_capture import XimeaCapture

    with pytest.raises(MissingDependencyError) as excinfo:
        XimeaCapture()

    err = excinfo.value
    assert isinstance(err, ImportError)
    message = str(err)
    assert "ximea" in message.lower()
    # No pip extra exists for Ximea; the message should instead point at the
    # vendor SDK download.
    assert "pip install" not in message
    assert "ximea.com" in message


def test_genicam_missing_harvesters_raises_missing_dependency_error(monkeypatch):
    _poison_import(monkeypatch, "harvesters")
    from framesource.sources.genicam_capture import GenicamCapture

    with pytest.raises(MissingDependencyError) as excinfo:
        GenicamCapture()

    err = excinfo.value
    assert isinstance(err, ImportError)
    message = str(err)
    assert "harvesters" in message
    assert "pip install framesource[genicam]" in message


def test_realsense_missing_pyrealsense2_raises_on_connect(monkeypatch):
    _poison_import(monkeypatch, "pyrealsense2")
    from framesource.sources.realsense_capture import RealsenseCapture

    # Construction never touches pyrealsense2 (imported lazily in connect()),
    # so it must still succeed without the SDK.
    cap = RealsenseCapture()

    with pytest.raises(MissingDependencyError) as excinfo:
        cap.connect()

    err = excinfo.value
    assert isinstance(err, ImportError)
    message = str(err)
    assert "pyrealsense2" in message
    assert "pip install framesource[realsense]" in message


def test_huateng_missing_mvsdk_raises_on_connect(monkeypatch):
    from framesource.sources import huateng_capture

    # The mvsdk SDK loads a vendor DLL at module-import time; simulate its
    # absence by patching the module-level handle directly rather than
    # reloading the module (which would require reconstructing import
    # poisoning for a relative `from . import mvsdk`).
    monkeypatch.setattr(huateng_capture, "mvsdk", None)
    monkeypatch.setattr(huateng_capture, "_MVSDK_IMPORT_ERROR", "No module named 'mvsdk'")

    cap = huateng_capture.HuatengCapture()

    with pytest.raises(MissingDependencyError) as excinfo:
        cap.connect()

    err = excinfo.value
    assert isinstance(err, ImportError)
    message = str(err)
    assert "mvsdk" in message
    # No pip extra exists for the Huateng/MindVision vendor SDK.
    assert "pip install" not in message
