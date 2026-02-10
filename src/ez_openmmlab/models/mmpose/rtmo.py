from pathlib import Path
from typing import Optional, Union

from loguru import logger
from mmengine.config import Config
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.core.engines.mmpose import EZMMPose
from ez_openmmlab.schemas.model import RTMO_CONFIGS, ModelName


class RTMO(EZMMPose):
    """RTMO implementation for one-stage, multi-person 2D pose estimation."""

    def __init__(
        self,
        model: Union[ModelName, str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
        **kwargs,
    ):
        """Initializes a new RTMO engine.

        RTMO (Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation)
        is a bottom-up model, meaning it detects people and their poses in a single
        pass without requiring a separate object detector.

        Args:
            model: The model variant to use. Supported: 'rtmo_s', 'rtmo_m', 'rtmo_l'.
                Can be a :class:`ModelName`, string, or path to a `config.toml`.
            checkpoint_path: Path to a custom model checkpoint (.pth). If None,
                the official weights will be downloaded automatically.
            log_level: Global logging level for the engine. Defaults to "INFO".
            **kwargs: Additional configuration parameters passed to the base engine.
        """
        self._validate_model(model)
        super().__init__(model, checkpoint_path, log_level, **kwargs)

    def _validate_model(self, model: Union[ModelName, str, Path]) -> None:
        """Validates that the provided model is a supported RTMO variant."""
        if isinstance(model, (str, Path)) and str(model).endswith(".toml"):
            return

        name = model.value if isinstance(model, ModelName) else str(model)
        if name not in RTMO_CONFIGS:
            supported = ", ".join(RTMO_CONFIGS.keys())
            raise ValueError(
                f"Invalid model variant '{name}' for RTMO. "
                f"Supported variants: {supported}, or a path to a custom config.toml"
            )

    def _instantiate_inferencer(
        self, cfg: Config, device: str, **kwargs
    ) -> MMPoseInferencer:
        """Instantiates the RTMO inferencer."""
        return MMPoseInferencer(
            pose2d=cfg,
            pose2d_weights=str(self.checkpoint_path),
            device=device,
        )