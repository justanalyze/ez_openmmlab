from pathlib import Path
from typing import Optional, Union

from ez_openmmlab.core.engines.mmdet import EZMMDetector
from ez_openmmlab.schemas.model import (
    RTM_DET_CONFIGS,
    RTM_DET_INS_CONFIGS,
    ModelName,
)


class RTMDet(EZMMDetector):
    """RTMDet implementation for fast object detection and instance segmentation.

    Args:
        model: Name of the RTMDet variant (e.g., 'rtmdet_tiny', 'rtmdet-ins_s') OR path to a config.toml.
        checkpoint_path: Path to a custom .pth file. If None, it will use the official pretrained weights.
        log_level: Global logging level for the internal engine ('INFO', 'WARNING', 'ERROR').
    """

    def __init__(
        self,
        model: Union[ModelName, str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
        **kwargs,
    ):
        self._validate_model(model)
        super().__init__(model, checkpoint_path, log_level, **kwargs)

    def _validate_model(self, model: Union[ModelName, str, Path]) -> None:
        """Validates that the provided model is a supported RTMDet variant."""
        if isinstance(model, (str, Path)) and str(model).endswith(".toml"):
            return

        name = model.value if isinstance(model, ModelName) else str(model)
        supported_configs = {**RTM_DET_CONFIGS, **RTM_DET_INS_CONFIGS}

        if name not in supported_configs:
            supported = ", ".join(supported_configs.keys())
            raise ValueError(
                f"Invalid model variant '{name}' for RTMDet. "
                f"Supported variants: {supported}, or a path to a custom config.toml"
            )