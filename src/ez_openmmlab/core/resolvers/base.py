from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class BaseModelParamsResolver(ABC):
    """Abstract base class for model-specific parameter resolution logic."""

    @abstractmethod
    def resolve(self, **kwargs) -> Dict[str, Any]:
        """Extracts, validates, and resolves parameters from user input.

        Args:
            **kwargs: Flexible keyword arguments (input_size, simcc_sigma, etc.)

        Returns:
            A dictionary of processed hyperparameters.
        """
        pass
