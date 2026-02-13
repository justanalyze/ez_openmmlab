from pathlib import Path
from typing import Optional, Union

from loguru import logger
from mmengine.config import Config
from mmdet.apis import DetInferencer
from mmdet.utils import register_all_modules

from ez_openmmlab.core.inference.formatters import DetectionResultFormatter
from ez_openmmlab.core.injectors.mmdet import MMDetInjector
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils.context import switch_to_lib_root
from ez_openmmlab.utils.toml_config import (
    DataSection,
    ModelSection,
    TrainingSection,
    UserConfig,
)

from .engine_base import EZMMLab

# Force registration of MMDet modules
register_all_modules(init_default_scope=True)


class EZMMDetector(EZMMLab):
    """Abstract base class for training and inference using MMDetection."""

    def __init__(
        self,
        model: Union[ModelName, str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
        **kwargs,
    ):
        super().__init__(model, checkpoint_path, log_level, **kwargs)
        self._inferencer: Optional[DetInferencer] = None
        self._formatter = DetectionResultFormatter()

    def _init_inferencer(self, device: str, **kwargs):
        """Lazy initialization of the DetInferencer with patching support."""
        if self._inferencer is None:
            logger.info(
                f"Initializing DetInferencer for model: {self.model} (using config: {self.config_path})"
            )
            det_cfg = self._load_and_patch_config()

            with switch_to_lib_root(self.model):
                self._inferencer = DetInferencer(
                    model=det_cfg,
                    weights=str(self.checkpoint_path),
                    device=device,
                )

    def _load_and_patch_config(self) -> Config:
        """Loads the detection config and applies runtime patches."""
        with switch_to_lib_root(self.model):
            cfg = Config.fromfile(str(self.config_path))

            # Only trigger patching if custom metadata is provided.
            # If num_classes is None, we assume the user wants to use the config's default.
            if self.num_classes is not None:
                dummy_user_cfg = self._get_dummy_user_config()
                MMDetInjector().apply(cfg, dummy_user_cfg)
            return cfg

    def _get_dummy_user_config(self) -> UserConfig:
        """Creates a dummy UserConfig to satisfy the injector interface."""
        return UserConfig(
            model=ModelSection(
                name=self.model,
                num_classes=self.num_classes,
            ),
            training=TrainingSection(num_workers=0, learning_rate=0.001),
            data=DataSection(root=""),
        )

    def _get_library_family(self) -> str:
        return "mmdet"

    def _run_inference(
        self, inputs: list, out_dir: str, show: bool, **kwargs
    ) -> Union[dict, list]:
        """Calls the DetInferencer with correct parameters."""
        # Map generic 'confidence' to mmdet 'pred_score_thr'
        confidence = kwargs.get("confidence") or kwargs.get("pred_score_thr", 0.3)

        assert self._inferencer is not None, "Inferencer not initialized."

        return self._inferencer(
            inputs, out_dir=out_dir, show=show, pred_score_thr=confidence
        )
