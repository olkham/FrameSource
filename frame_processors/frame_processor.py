from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any


class FrameProcessor(ABC):
    """Base class for all frame processors."""
    
    def __init__(self):
        self._parameters = {}
        
    @abstractmethod
    def process(self, frame: np.ndarray) -> np.ndarray:
        """Process a frame and return the processed frame."""
        pass
    
    def set_parameter(self, name: str, value: Any) -> None:
        """Set a processing parameter."""
        self._parameters[name] = value
        
    def get_parameter(self, name: str) -> Any:
        """Get a processing parameter."""
        return self._parameters.get(name)
    
    def get_parameters(self) -> Dict[str, Any]:
        """Get all processing parameters."""
        return self._parameters.copy()

