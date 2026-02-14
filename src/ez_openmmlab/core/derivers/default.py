from typing import Any, Dict, Optional, Tuple

from .base import BaseParameterDeriver


class DefaultParameterDeriver(BaseParameterDeriver):
    """Fallback derivation logic for standard models."""

    def derive(
        self,
        input_size: Tuple[int, int],
        simcc_sigma: Optional[Tuple[float, float]],
        feature_map_size: Optional[Tuple[int, int]],
    ) -> Dict[str, Any]:
        return {
            "input_size": input_size,
            "simcc_sigma": simcc_sigma or (4.9, 5.66),
            "feature_map_size": feature_map_size,
        }
