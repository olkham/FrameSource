"""Tests for the typed constructors (Step 7.2) and get_config_schema deprecation.

These verify that promoting previously ``**kwargs``-mined options to explicit
keyword parameters keeps full backwards compatibility: the old ``**kwargs``
call style must behave identically, unknown kwargs must still flow into
``self.config``, and the promoted names must be visible on the signature.
"""

import inspect

import pytest

from framesource.sources.webcam_capture import WebcamCapture
from framesource.sources.video_file_capture import VideoFileCapture
from framesource.sources.folder_capture import FolderCapture
from framesource.sources.ipcamera_capture import IPCameraCapture
from framesource.sources.screen_capture import ScreenCapture


def _param_names(cls):
    return set(inspect.signature(cls.__init__).parameters)


# --------------------------------------------------------------------------- #
# WebcamCapture
# --------------------------------------------------------------------------- #
def test_webcam_named_params_populate_config():
    cap = WebcamCapture(source=0, width=1280, height=720, fps=60)
    assert cap.source == 0
    assert cap.config["width"] == 1280
    assert cap.config["height"] == 720
    assert cap.config["fps"] == 60


def test_webcam_kwargs_style_is_identical():
    named = WebcamCapture(source=0, width=1280, height=720, fps=60)
    kwargs = WebcamCapture(**{"source": 0, "width": 1280, "height": 720, "fps": 60})
    assert named.config == kwargs.config


def test_webcam_absent_optional_params_not_in_config():
    cap = WebcamCapture(0)
    assert "width" not in cap.config
    assert "height" not in cap.config
    assert "fps" not in cap.config


def test_webcam_unknown_kwarg_lands_in_config():
    cap = WebcamCapture(0, some_future_option=123)
    assert cap.config["some_future_option"] == 123


def test_webcam_signature_exposes_promoted_params():
    assert {"width", "height", "fps"} <= _param_names(WebcamCapture)


# --------------------------------------------------------------------------- #
# VideoFileCapture
# --------------------------------------------------------------------------- #
def test_videofile_named_params_set_attrs_and_config():
    cap = VideoFileCapture("clip.mp4", loop=True, real_time=False,
                           width=640, height=480, fps=25)
    assert cap.loop is True
    assert cap.real_time is False
    assert cap.config["width"] == 640
    assert cap.config["height"] == 480
    assert cap.config["fps"] == 25


def test_videofile_kwargs_style_is_identical():
    named = VideoFileCapture("clip.mp4", loop=True, real_time=False, fps=25)
    kwargs = VideoFileCapture("clip.mp4", **{"loop": True, "real_time": False, "fps": 25})
    assert named.loop == kwargs.loop
    assert named.real_time == kwargs.real_time
    assert named.config == kwargs.config


def test_videofile_defaults_match_old_behaviour():
    cap = VideoFileCapture("clip.mp4")
    assert cap.loop is False
    assert cap.real_time is True


def test_videofile_unknown_kwarg_lands_in_config():
    cap = VideoFileCapture("clip.mp4", exotic=1)
    assert cap.config["exotic"] == 1


def test_videofile_signature_exposes_promoted_params():
    assert {"loop", "real_time", "width", "height", "fps"} <= _param_names(VideoFileCapture)


# --------------------------------------------------------------------------- #
# FolderCapture (already had positional params; verify preserved + typed)
# --------------------------------------------------------------------------- #
def test_folder_named_params_set_attrs():
    cap = FolderCapture("some/folder", sort_by="date", fps=10,
                        loop=True, real_time=False, watch_folder=False)
    assert cap.sort_by == "date"
    assert cap.fps == 10
    assert cap.loop is True
    assert cap.real_time is False
    assert cap.watch_folder is False


def test_folder_positional_sort_by_preserved():
    cap = FolderCapture("some/folder", "date")
    assert cap.sort_by == "date"


def test_folder_unknown_kwarg_lands_in_config():
    cap = FolderCapture("some/folder", exotic="x")
    assert cap.config["exotic"] == "x"


def test_folder_signature_exposes_params():
    assert {"sort_by", "width", "height", "fps", "real_time", "loop",
            "watch_folder"} <= _param_names(FolderCapture)


# --------------------------------------------------------------------------- #
# IPCameraCapture
# --------------------------------------------------------------------------- #
def test_ipcamera_named_params_set_attrs_and_config():
    cap = IPCameraCapture("rtsp://host/stream", username="u", password="p",
                          width=800, height=600, fps=25)
    assert cap.username == "u"
    assert cap.password == "p"
    assert cap.config["width"] == 800
    assert cap.config["fps"] == 25
    assert cap.stream_url == "rtsp://u:p@host/stream"


def test_ipcamera_positional_credentials_preserved():
    cap = IPCameraCapture("rtsp://host/stream", "u", "p")
    assert cap.username == "u"
    assert cap.password == "p"


def test_ipcamera_kwargs_style_is_identical():
    named = IPCameraCapture("rtsp://host/stream", width=800, fps=25)
    kwargs = IPCameraCapture("rtsp://host/stream", **{"width": 800, "fps": 25})
    assert named.config == kwargs.config


def test_ipcamera_unknown_kwarg_lands_in_config():
    cap = IPCameraCapture("rtsp://host/stream", exotic=1)
    assert cap.config["exotic"] == 1


def test_ipcamera_signature_exposes_promoted_params():
    assert {"username", "password", "width", "height", "fps"} <= _param_names(IPCameraCapture)


# --------------------------------------------------------------------------- #
# ScreenCapture
# --------------------------------------------------------------------------- #
def test_screen_named_params_set_attrs():
    cap = ScreenCapture(x=10, y=20, w=100, h=50, fps=15, hwnd=999)
    assert (cap.x, cap.y, cap.w, cap.h) == (10, 20, 100, 50)
    assert cap.fps == 15
    assert cap.hwnd == 999
    assert cap.config["hwnd"] == 999


def test_screen_hwnd_kwargs_style_is_identical():
    named = ScreenCapture(hwnd=5)
    kwargs = ScreenCapture(**{"hwnd": 5})
    assert named.hwnd == kwargs.hwnd == 5
    assert named.config == kwargs.config


def test_screen_absent_hwnd_not_in_config():
    cap = ScreenCapture()
    assert cap.hwnd is None
    assert "hwnd" not in cap.config


def test_screen_unknown_kwarg_lands_in_config():
    cap = ScreenCapture(exotic="x")
    assert cap.config["exotic"] == "x"


def test_screen_signature_exposes_params():
    assert {"x", "y", "w", "h", "fps", "hwnd"} <= _param_names(ScreenCapture)


# --------------------------------------------------------------------------- #
# AudioSpectrogramCapture (needs optional audio deps)
# --------------------------------------------------------------------------- #
def _audio_cls():
    pytest.importorskip("librosa")
    pytest.importorskip("soundfile")
    pytest.importorskip("pyaudio")
    from framesource.sources.audiospectrogram_capture import AudioSpectrogramCapture
    return AudioSpectrogramCapture


def test_audio_named_params_set_attrs():
    AudioSpectrogramCapture = _audio_cls()
    cap = AudioSpectrogramCapture(source=0, n_mels=64, n_fft=1024, frame_rate=15,
                                  contrast_method="percentile")
    assert cap.n_mels == 64
    assert cap.n_fft == 1024
    assert cap.frame_rate == 15
    assert cap.contrast_method == "percentile"


def test_audio_kwargs_style_is_identical():
    AudioSpectrogramCapture = _audio_cls()
    named = AudioSpectrogramCapture(source=0, n_mels=64, frame_rate=15)
    kwargs = AudioSpectrogramCapture(source=0, **{"n_mels": 64, "frame_rate": 15})
    assert named.n_mels == kwargs.n_mels
    assert named.frame_rate == kwargs.frame_rate


def test_audio_unknown_kwarg_lands_in_config():
    AudioSpectrogramCapture = _audio_cls()
    cap = AudioSpectrogramCapture(source=0, exotic="x")
    assert cap.config["exotic"] == "x"


def test_audio_signature_exposes_promoted_params():
    AudioSpectrogramCapture = _audio_cls()
    assert {"n_mels", "n_fft", "hop_length", "window_duration", "freq_range",
            "sample_rate", "colormap", "frame_rate"} <= _param_names(AudioSpectrogramCapture)


# --------------------------------------------------------------------------- #
# get_config_schema() deprecation (Step 7.11)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("cls", [
    WebcamCapture, VideoFileCapture, FolderCapture, IPCameraCapture, ScreenCapture,
])
def test_get_config_schema_deprecated(cls):
    with pytest.warns(DeprecationWarning):
        schema = cls.get_config_schema()
    # Payload must be preserved untouched.
    assert "fields" in schema
