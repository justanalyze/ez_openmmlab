from pathlib import Path
from typing import Optional, Union, List

from loguru import logger
from mmengine.config import Config
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.core.engines.mmpose import EZMMPose
from ez_openmmlab.core.config_manager import get_config_file
from ez_openmmlab.core.results import InferenceResult
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils.download import ensure_model_checkpoint
from ez_openmmlab.utils.toml_config import (
    UserConfig,
    ModelSection,
    DataSection,
    TrainingSection,
)
from ez_openmmlab.core.injectors.mmpose import MMPoseInjector
from ez_openmmlab.utils.context import switch_to_lib_root


class RTMPose(EZMMPose):
    """RTMPose implementation for fast 2D keypoint estimation.

    Supported variants: rtmpose_tiny, rtmpose_s, rtmpose_m, rtmpose_l.
    Note: RTMPose is a top-down model and requires a detector.
    """

    def __init__(
        self,
        model: Union[ModelName, str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
    ):
        super().__init__(model, checkpoint_path, log_level)
        self._inferencer: Optional[MMPoseInferencer] = None

    def _init_inferencer(self, device: str, **kwargs):
        """Lazy initialization of the RTMPose inferencer."""
        if self._inferencer is not None:
            return

        # 1. Resolve detector components (Stage 1)
        det_config, det_weights, det_cat_ids = self._resolve_detector_params(kwargs)

        # 2. Prepare and patch the pose config (Stage 2)
        pose_cfg = self._load_and_patch_config()

        logger.info(f"Initializing RTMPose inferencer: {self.model}")

        # 3. Instantiate the inferencer
        with switch_to_lib_root(self.model):
            self._inferencer = MMPoseInferencer(
                pose2d=pose_cfg,
                pose2d_weights=str(self.checkpoint_path),
                det_model=det_config,
                det_weights=det_weights,
                det_cat_ids=det_cat_ids,
                device=device,
            )

    def _resolve_detector_params(self, kwargs: dict) -> tuple:
        """Resolves detector config, weights, and category IDs."""
        det_model = kwargs.get("det_model", "rtmdet_tiny")
        det_weights = kwargs.get("det_weights", None)
        det_cat_ids = kwargs.get("det_cat_ids", [0])

        if det_model in [m.value for m in ModelName]:
            det_config = str(get_config_file(det_model))
            if not det_weights:
                det_weights = str(ensure_model_checkpoint(det_model))
        else:
            det_config = det_model

        return det_config, det_weights, det_cat_ids

    def _load_and_patch_config(self) -> Config:
        """Loads the pose config and applies runtime patches using the plugin system."""
        with switch_to_lib_root(self.model):
            cfg = Config.fromfile(str(self.config_path))

            # Wrap current state into a dummy UserConfig to reuse MMPoseInjector logic
            dummy_user_cfg = UserConfig(
                model=ModelSection(
                    name=self.model,
                    num_classes=self.num_classes
                    if self.num_classes is not None
                    else 80,  # Dummy valid value
                    num_keypoints=self.num_keypoints,
                ),
                training=TrainingSection(num_workers=0, learning_rate=0.001),
                data=DataSection(root=""),
            )
            # If both are None, MMPoseInjector won't patch anything (which we want for standard models)
            if self.num_classes is not None or self.num_keypoints is not None:
                MMPoseInjector().apply(cfg, dummy_user_cfg)
            return cfg

