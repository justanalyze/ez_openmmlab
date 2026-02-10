from pathlib import Path
from typing import Optional, Union

from ez_openmmlab.core.engines.mmdet import EZMMDetector
from ez_openmmlab.schemas.model import (
    RTM_DET_CONFIGS,
    RTM_DET_INS_CONFIGS,
    ModelName,
)


class RTMDet(EZMMDetector):
    """RTMDet implementation for high-speed object detection and instance segmentation.

    RTMDet (Real-time Models for Object Detection) is a family of high-efficiency
    models from MMDetection. This class provides a simplified interface for
    training and inference using these models.

    Args:
        model: The model variant to use. Can be a member of :class:`ModelName`,
            a string name (e.g., 'rtmdet_tiny'), or a path to a custom `config.toml`.
        checkpoint_path: Path to a custom model checkpoint (.pth). If None,
            the official pretrained weights will be automatically downloaded
            based on the `model` name.
        log_level: Global logging level for the engine. Defaults to "INFO".
        **kwargs: Additional configuration parameters passed to the base engine.

    Attributes:
        model (str): The resolved model name or path.
        checkpoint_path (Path): Absolute path to the resolved weights file.
        num_classes (int, optional): Number of output classes for the model.
    """

    def __init__(
        self,
        model: Union[ModelName, str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
    ):
        self._validate_model(model)
        super().__init__(model, checkpoint_path, log_level)

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
