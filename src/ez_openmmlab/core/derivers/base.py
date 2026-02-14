from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class BaseParameterDeriver(ABC):
    """Abstract base class for model-specific parameter derivation logic."""

    @abstractmethod
    def derive(self, **kwargs) -> Dict[str, Any]:
        """Extracts, validates, and derives parameters from user input.

        Args:
            **kwargs: Flexible keyword arguments (input_size, simcc_sigma, etc.)

        Returns:
            A dictionary of processed hyperparameters.
        """
        pass
