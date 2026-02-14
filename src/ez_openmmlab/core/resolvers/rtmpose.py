from typing import Any, Dict, Optional, Tuple

from loguru import logger

from .base import BaseModelParamsResolver


class RTMPoseParamsResolver(BaseModelParamsResolver):
    """Parameter resolution logic specific to the RTMPose family."""

    def resolve(self, **kwargs) -> Dict[str, Any]:
        """Processes RTMPose specific parameters with auto-adjustment logic."""
        # 1. Extraction with defaults
        input_size = kwargs.get("input_size") or (192, 256)
        simcc_sigma = kwargs.get("simcc_sigma")
        feature_map_size = kwargs.get("feature_map_size")

        # 2. Input Size Auto-Adjustment (32-divisibility recommender)
        adjusted_input_size = self._adjust_input_size(input_size)

        # 3. Derive Sigma (Linear scaling based on 192x256 reference)
        if simcc_sigma is None:
            simcc_sigma = self._scale_sigma(adjusted_input_size)

        # 4. Derive Feature Map Size (stride 32 default)
        if feature_map_size is None:
            feature_map_size = (
                adjusted_input_size[0] // 32,
                adjusted_input_size[1] // 32,
            )
            logger.debug(f"RTMPose: Auto-derived feature_map_size: {feature_map_size}")

        # 5. Validation
        self._validate_compatibility(adjusted_input_size, simcc_sigma, feature_map_size)

        return {
            "input_size": adjusted_input_size,
            "simcc_sigma": simcc_sigma,
            "feature_map_size": feature_map_size,
        }

    def _adjust_input_size(self, input_size: Tuple[int, int]) -> Tuple[int, int]:
        """Ensures dimensions are divisible by 32. Warns and adjusts if not."""
        w, h = input_size
        new_w = self._round_to_32(w)
        new_h = self._round_to_32(h)

        if (new_w, new_h) != input_size:
            logger.warning(
                f"RTMPose architectural constraint: 'input_size' should be divisible by 32. "
                f"Automatically adjusting {input_size} -> {(new_w, new_h)}."
            )
        return (new_w, new_h)

    def _round_to_32(self, val: int) -> int:
        """Rounds to the nearest multiple of 32."""
        if val % 32 == 0:
            return val
        return int(round(val / 32) * 32)

    def _scale_sigma(self, input_size: Tuple[int, int]) -> Tuple[float, float]:
        if input_size == (192, 256):
            return (4.9, 5.66)
        
        scale_w = input_size[0] / 192
        scale_h = input_size[1] / 256
        sigma = (round(4.9 * scale_w, 2), round(5.66 * scale_h, 2))
        logger.info(f"RTMPose: Auto-scaling simcc_sigma to {sigma} for input_size {input_size}")
        return sigma

    def _validate_compatibility(
        self, 
        input_size: Tuple[int, int], 
        sigma: Tuple[float, float], 
        feature_map: Tuple[int, int]
    ) -> None:
        max_sigma = max(sigma)
        min_dim = min(input_size)
        if max_sigma > min_dim / 4:
             logger.warning(
                 f"RTMPose: Large simcc_sigma {sigma} detected for input_size {input_size}. "
                 "This may lead to poor convergence."
             )
