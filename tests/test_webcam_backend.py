"""Hardware-free tests for WebcamCapture backend/fourcc configuration.

These exercise the construction-time resolution logic only (no camera is
opened), so they run anywhere.
"""

import platform

import cv2
import pytest

from framesource.sources.webcam_capture import (
    _BACKEND_ALIASES,
    _UNCOMPRESSED_FOURCCS,
    WebcamCapture,
)


def test_default_backend_matches_platform():
    cam = WebcamCapture(source=0)
    expected = {
        "Windows": cv2.CAP_DSHOW,
        "Darwin": cv2.CAP_AVFOUNDATION,
    }.get(platform.system(), cv2.CAP_V4L2)
    assert cam.api_preference == expected


@pytest.mark.parametrize("name", sorted(_BACKEND_ALIASES))
def test_backend_name_resolves(name):
    cam = WebcamCapture(source=0, backend=name)
    assert cam.api_preference == _BACKEND_ALIASES[name]


def test_backend_name_is_case_insensitive():
    assert WebcamCapture(source=0, backend="MSMF").api_preference == _BACKEND_ALIASES["msmf"]
    assert WebcamCapture(source=0, backend=" DShow ").api_preference == _BACKEND_ALIASES["dshow"]


def test_backend_int_passthrough():
    assert WebcamCapture(source=0, backend=cv2.CAP_MSMF).api_preference == cv2.CAP_MSMF


def test_unknown_backend_raises_value_error():
    with pytest.raises(ValueError, match="Unknown webcam backend"):
        WebcamCapture(source=0, backend="not-a-backend")


def test_fourcc_is_stored_and_defaults_to_none():
    assert WebcamCapture(source=0).fourcc is None
    assert WebcamCapture(source=0, fourcc="MJPG").fourcc == "MJPG"


def test_yuy2_is_flagged_uncompressed():
    # Guards the diagnostic that warns about bandwidth-limited high-res streams.
    assert "YUY2" in _UNCOMPRESSED_FOURCCS
    assert "MJPG" not in _UNCOMPRESSED_FOURCCS
