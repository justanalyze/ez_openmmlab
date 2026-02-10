from pathlib import Path
from typing import Optional, Union

from loguru import logger
from mmengine.config import Config
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.core.engines.mmpose import EZMMPose
from ez_openmmlab.core.injectors.mmpose import MMPoseInjector
from ez_openmmlab.schemas.model import RTMO_CONFIGS, ModelName
from ez_openmmlab.utils.context import switch_to_lib_root
from ez_openmmlab.utils.toml_config import (
    DataSection,
    ModelSection,
    TrainingSection,
    UserConfig,
)


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
        self._inferencer: Optional[MMPoseInferencer] = None

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

    def _init_inferencer(self, device: str, **kwargs):
        """Lazy initialization of the RTMO inferencer."""
        if self._inferencer is None:
            logger.info(f"Initializing RTMO inferencer: {self.model}")
            with switch_to_lib_root(self.model):
                cfg = Config.fromfile(str(self.config_path))

                # Apply pose-specific patches using the injector
                dummy_user_cfg = UserConfig(
                    model=ModelSection(
                        name=self.model,
                        num_classes=self.num_classes
                        if self.num_classes is not None
                        else 80,
                        num_keypoints=self.num_keypoints,
                    ),
                    training=TrainingSection(num_workers=0, learning_rate=0.001),
                    data=DataSection(root=""),
                )
                if self.num_classes is not None or self.num_keypoints is not None:
                    MMPoseInjector().apply(cfg, dummy_user_cfg)

                self._inferencer = MMPoseInferencer(
                    pose2d=cfg,
                    pose2d_weights=str(self.checkpoint_path),
                    device=device,
                )
