from abc import ABC, abstractmethod
from typing import Any, Dict, Optional, Tuple


class BaseParameterDeriver(ABC):
    """Abstract base class for model-specific parameter derivation logic."""

    @abstractmethod
    def derive(
        self,
        input_size: Tuple[int, int],
        simcc_sigma: Optional[Tuple[float, float]],
        feature_map_size: Optional[Tuple[int, int]],
    ) -> Dict[str, Any]:
        """Derives and validates parameters.

        Returns:
            A dictionary containing adjusted 'input_size', 'simcc_sigma', and 'feature_map_size'.
        """
        pass
