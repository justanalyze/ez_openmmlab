from pathlib import Path
from typing import Optional, Union

from mmengine.config import Config
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.core.engines.mmpose import EZMMPose
from ez_openmmlab.schemas.model import RTMO_CONFIGS, ModelName


class RTMO(EZMMPose):
    """RTMO implementation for fast bottom-up 2D multi-person pose estimation.

    Supported variants: rtmo_s, rtmo_m, rtmo_l.
    Note: RTMO is a bottom-up model and does not require a separate detector.
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