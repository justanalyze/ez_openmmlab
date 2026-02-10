from pathlib import Path
from typing import List, Optional, Union

from mmengine.config import Config
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.core.config_manager import get_config_file
from ez_openmmlab.core.engines.mmpose import EZMMPose
from ez_openmmlab.schemas.model import RTM_POSE_CONFIGS, ModelName
from ez_openmmlab.utils.download import ensure_model_checkpoint


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
        det_model: Union[ModelName, str, Path] = "rtmdet_tiny",
        det_weights: Optional[Union[str, Path]] = None,
        det_cat_ids: List[int] = [0],
        **kwargs,
    ):
        self._validate_model(model)
        super().__init__(model, checkpoint_path, log_level, **kwargs)
        self.det_model = det_model
        self.det_weights = det_weights
        self.det_cat_ids = det_cat_ids

    def _validate_model(self, model: Union[ModelName, str, Path]) -> None:
        """Validates that the provided model is a supported RTMPose variant."""
        if isinstance(model, (str, Path)) and str(model).endswith(".toml"):
            return

        name = model.value if isinstance(model, ModelName) else str(model)
        if name not in RTM_POSE_CONFIGS:
            supported = ", ".join(RTM_POSE_CONFIGS.keys())
            raise ValueError(
                f"Invalid model variant '{name}' for RTMPose. "
                f"Supported variants: {supported}, or a path to a custom config.toml"
            )

    def _instantiate_inferencer(
        self, cfg: Config, device: str, **kwargs
    ) -> MMPoseInferencer:
        """Instantiates the RTMPose inferencer with detector resolution."""
        # 1. Resolve detector components (Stage 1)
        det_config, det_weights, det_cat_ids = self._resolve_detector_params()

        # 2. Instantiate
        return MMPoseInferencer(
            pose2d=cfg,
            pose2d_weights=str(self.checkpoint_path),
            det_model=det_config,
            det_weights=det_weights,
            det_cat_ids=det_cat_ids,
            device=device,
        )

    def _resolve_detector_params(self) -> tuple:
        """Resolves detector config, weights, and category IDs."""
        if self.det_model in [m.value for m in ModelName]:
            det_config = str(get_config_file(self.det_model))
            det_weights = self.det_weights
            if not det_weights:
                det_weights = str(ensure_model_checkpoint(self.det_model))
        else:
            det_config = str(self.det_model)
            det_weights = str(self.det_weights) if self.det_weights else None

        return det_config, det_weights, self.det_cat_ids
