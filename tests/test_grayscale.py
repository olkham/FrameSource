"""Colormap parameter handling for AudioSpectrogramCapture (no hardware required).

Constructs the capture without connecting, so no microphone or GUI is needed.
"""

import pytest

pytest.importorskip("librosa")
pytest.importorskip("soundfile")
pytest.importorskip("pyaudio")

import cv2

from framesource.sources.audiospectrogram_capture import AudioSpectrogramCapture


def make_capture(**kwargs):
    return AudioSpectrogramCapture(source=None, n_mels=64, window_duration=1.0,
                                   frame_rate=10, **kwargs)


def test_default_colormap_is_grayscale():
    camera = make_capture()
    assert camera.get_colormap() is None


def test_set_colormap_switches():
    camera = make_capture()
    assert camera.set_colormap(cv2.COLORMAP_VIRIDIS)
    assert camera.get_colormap() == cv2.COLORMAP_VIRIDIS
    assert camera.set_colormap(None)
    assert camera.get_colormap() is None


def test_colormap_accepted_at_construction():
    camera = make_capture(colormap=cv2.COLORMAP_JET)
    assert camera.get_colormap() == cv2.COLORMAP_JET
