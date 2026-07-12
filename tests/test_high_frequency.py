"""Frequency-range handling for AudioSpectrogramCapture (no hardware required).

Covers `_parse_freq_range` (importable without audio deps) and the automatic
sample-rate adjustment when a range exceeds the current Nyquist limit.
"""

import pytest

from framesource.sources.audiospectrogram_capture import _parse_freq_range


class TestParseFreqRange:
    def test_tuple(self):
        assert _parse_freq_range((20, 8000)) == (20.0, 8000.0)

    def test_list(self):
        assert _parse_freq_range([20, 8000]) == (20.0, 8000.0)

    def test_string(self):
        assert _parse_freq_range("20,8000") == (20.0, 8000.0)

    @pytest.mark.parametrize("bad", [(20,), (20, 100, 200), "20", "20,30,40", 42, None, ("a", "b")])
    def test_invalid_raises(self, bad):
        with pytest.raises(ValueError):
            _parse_freq_range(bad)


class TestHighFrequencyConfig:
    @pytest.fixture(autouse=True)
    def _require_audio_deps(self):
        pytest.importorskip("librosa")
        pytest.importorskip("soundfile")
        pytest.importorskip("pyaudio")

    def make_capture(self, **kwargs):
        from framesource.sources.audiospectrogram_capture import AudioSpectrogramCapture

        return AudioSpectrogramCapture(
            source=None, n_mels=256, window_duration=2.0, frame_rate=30, **kwargs
        )

    def test_tuple_freq_range_accepted(self):
        camera = self.make_capture(freq_range=(20, 8000))
        assert camera.get_freq_range() == (20.0, 8000.0)

    def test_sample_rate_auto_adjusts_for_full_audible_range(self):
        camera = self.make_capture(freq_range=(20, 20000))
        # 20 kHz needs a sample rate of at least 44 kHz (2.2x margin) -> 44100.
        assert camera.get_sample_rate() == 44100
        assert camera.get_nyquist_frequency() >= 20000
        is_valid, message = camera.validate_frequency_range(20, 20000)
        assert is_valid, message

    def test_validate_frequency_range_rejects_bad_input(self):
        camera = self.make_capture(freq_range=(20, 8000))
        assert not camera.validate_frequency_range(0, 8000)[0]
        assert not camera.validate_frequency_range(100, 50)[0]
        # Beyond Nyquist for the default 44100 Hz sample rate.
        assert not camera.validate_frequency_range(20, 30000)[0]
