from pathlib import Path
from typing import Optional, Union, List

from loguru import logger
from mmengine.config import Config
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.engines.mmpose import EZMMPose
from ez_openmmlab.core.config_loader import get_config_file
from ez_openmmlab.core.results import InferenceResult
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils.toml_config import UserConfig, ModelSection, TrainingSection, DataSection
from ez_openmmlab.core.injectors.mmpose import MMPoseInjector
from ez_openmmlab.utils.context import switch_to_lib_root


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
        super().__init__(model, checkpoint_path, log_level)
        self._inferencer: Optional[MMPoseInferencer] = None

    def predict(
        self,
        image_path: Union[str, Path, list],
        *,
        bbox_thr: float = 0.3,
        kpt_thr: float = 0.3,
        device: str = "cuda",
        show: bool = False,
        out_dir: Optional[str] = None,
        **kwargs,
    ) -> List[InferenceResult]:
        """Runs RTMO inference

        Args:
            image_path: Path to a single image, a list of paths, or a directory.
            bbox_thr: Bounding box score threshold.
            kpt_thr: Keypoint score threshold.
            device: Computing device ('cuda', 'cpu').
            show: Whether to display results.
            out_dir: Directory to save visualization.
            det_cat_ids: Category IDs (not typically used by bottom-up RTMO).
        """
        return super().predict(
            image_path=image_path,
            bbox_thr=bbox_thr,
            kpt_thr=kpt_thr,
            device=device,
            show=show,
            out_dir=out_dir,
            **kwargs,
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
                        num_classes=self.num_classes if self.num_classes is not None else 80,
                        num_keypoints=self.num_keypoints
                    ),
                    training=TrainingSection(num_workers=0, learning_rate=0.001),
                    data=DataSection(root="")
                )
                if self.num_classes is not None or self.num_keypoints is not None:
                    MMPoseInjector().apply(cfg, dummy_user_cfg)

                self._inferencer = MMPoseInferencer(
                    pose2d=cfg,
                    pose2d_weights=str(self.checkpoint_path),
                    device=device,
                )
