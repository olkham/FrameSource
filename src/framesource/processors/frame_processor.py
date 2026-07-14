from abc import ABC, abstractmethod
from typing import Any, Union

import numpy as np

FrameType = Union[np.ndarray, dict[str, np.ndarray]]


class FrameProcessor(ABC):
    """Base class for all frame processors."""

    def __init__(self):
        self._parameters = {}

    @abstractmethod
    def process(self, frame: FrameType) -> FrameType:
        """Process a frame and return the processed frame."""
        pass

    def set_parameter(self, name: str, value: Any) -> None:
        """Set a processing parameter."""
        self._parameters[name] = value

    def get_parameter(self, name: str) -> Any:
        """Get a processing parameter."""
        return self._parameters.get(name)

    def get_parameters(self) -> dict[str, Any]:
        """Get all processing parameters."""
        return self._parameters.copy()
