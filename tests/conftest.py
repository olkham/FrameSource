"""Shared pytest fixtures and a hardware-free MockCapture for FrameSource tests."""

from typing import Optional

import numpy as np
import pytest

from framesource.sources.video_capture_base import VideoCaptureBase


class MockCapture(VideoCaptureBase):
    """An in-memory capture source with no hardware dependencies.

    Produces deterministic frames (a solid color that increments each read) so
    tests can exercise the capture contract, Frame metadata, and threading
    helpers without a real device.
    """

    has_discovery = False

    def __init__(
        self,
        source="mock",
        width: int = 16,
        height: int = 12,
        max_frames: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(source, **kwargs)
        self.width = width
        self.height = height
        self.max_frames = max_frames
        self._reads = 0
        self._exposure = 0.0
        self._gain = 0.0
        self._auto_exposure = False

    def connect(self) -> bool:
        self.is_connected = True
        return True

    def disconnect(self) -> bool:
        self.is_connected = False
        return True

    def _read_implementation(self) -> tuple[bool, Optional[np.ndarray]]:
        if not self.is_connected:
            return False, None
        if self.max_frames is not None and self._reads >= self.max_frames:
            return False, None
        # Deterministic frame: filled with the current read index (mod 256).
        value = self._reads % 256
        frame = np.full((self.height, self.width, 3), value, dtype=np.uint8)
        self._reads += 1
        return True, frame

    def enable_auto_exposure(self, enable: bool = True) -> bool:
        self._auto_exposure = enable
        return True

    def set_exposure(self, value: float) -> bool:
        self._exposure = value
        return True

    def get_exposure(self) -> Optional[float]:
        return self._exposure

    def set_gain(self, value: float) -> bool:
        self._gain = value
        return True

    def get_gain(self) -> Optional[float]:
        return self._gain

    def get_frame_size(self) -> Optional[tuple[int, int]]:
        return (self.width, self.height)


@pytest.fixture
def mock_capture():
    """Return a fresh, unconnected MockCapture instance."""
    return MockCapture()
