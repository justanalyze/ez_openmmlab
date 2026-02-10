from pathlib import Path
from typing import Optional, Union

from loguru import logger
from mmengine.config import Config
from mmdet.apis import DetInferencer

from ez_openmmlab.core.engines.mmdet import EZMMDetector
from ez_openmmlab.core.injectors.mmdet import MMDetInjector
from ez_openmmlab.schemas.model import (
    RTM_DET_CONFIGS,
    RTM_DET_INS_CONFIGS,
    ModelName,
)
from ez_openmmlab.utils.context import switch_to_lib_root
from ez_openmmlab.utils.toml_config import (
    DataSection,
    ModelSection,
    TrainingSection,
    UserConfig,
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

    def _init_inferencer(self, device: str, **kwargs):
        """Lazy initialization of the DetInferencer with patching support."""
        if self._inferencer is None:
            logger.info(
                f"Initializing DetInferencer for model: {self.model} (using config: {self.config_path})"
            )
            det_cfg = self._load_and_patch_config()

            self._inferencer = DetInferencer(
                model=det_cfg,
                weights=str(self.checkpoint_path),
                device=device,
            )

    def _load_and_patch_config(self) -> Config:
        """Loads the detection config and applies runtime patches."""
        with switch_to_lib_root(self.model):
            cfg = Config.fromfile(str(self.config_path))

            if self.num_classes is not None:
                dummy_user_cfg = self._get_dummy_user_config()
                MMDetInjector().apply(cfg, dummy_user_cfg)
            return cfg

    def _get_dummy_user_config(self) -> UserConfig:
        """Creates a dummy UserConfig to satisfy the injector interface."""
        return UserConfig(
            model=ModelSection(
                name=self.model,
                num_classes=self.num_classes if self.num_classes is not None else 80,
            ),
            training=TrainingSection(num_workers=0, learning_rate=0.001),
            data=DataSection(root=""),
        )
