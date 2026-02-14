from typing import Any, Dict, Optional, Tuple

from .base import BaseParameterDeriver


class DefaultParameterDeriver(BaseParameterDeriver):
    """Fallback derivation logic for standard models."""

    def derive(self, **kwargs) -> Dict[str, Any]:
        return {
            "input_size": kwargs.get("input_size") or (640, 640),
            "simcc_sigma": kwargs.get("simcc_sigma") or (4.9, 5.66),
            "feature_map_size": kwargs.get("feature_map_size"),
        }
