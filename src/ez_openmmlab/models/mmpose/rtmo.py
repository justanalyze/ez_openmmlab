from pathlib import Path
from typing import Optional, Union, List

from loguru import logger
from mmengine.config import Config
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.engines.mmpose import EZMMPose
from ez_openmmlab.core.config_loader import get_config_file
from ez_openmmlab.core.results import InferenceResult
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils.toml_config import UserConfig


class RTMO(EZMMPose):
    """RTMO implementation for fast bottom-up 2D multi-person pose estimation.

    Supported variants: rtmo_s, rtmo_m, rtmo_l.
    Note: RTMO is a bottom-up model and does not require a separate detector.
    """

    def __init__(
        self,
        model_name: ModelName | str,
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
    ):
        super().__init__(model_name, checkpoint_path, log_level)
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
            config_path = get_config_file(self.model_name)

            logger.info(f"Initializing RTMO inferencer: {self.model_name}")
            with self.switch_to_lib_root():
                # Load config to check if we need to override keypoints
                cfg = Config.fromfile(str(config_path))

                if self.num_keypoints:
                    logger.info(
                        f"Overriding model.head.num_keypoints to {self.num_keypoints} for inference"
                    )
                    if hasattr(cfg.model, "head"):
                        cfg.model.head.num_keypoints = self.num_keypoints

                self._inferencer = MMPoseInferencer(
                    pose2d=cfg,
                    pose2d_weights=str(self.checkpoint_path),
                    device=device,
                )

    def _configure_model_specifics(self, config: UserConfig) -> None:
        """RTMO specific head overrides."""
        if not self._cfg:
            raise RuntimeError("Config not loaded.")

        if hasattr(self._cfg.model, "head"):
            head = self._cfg.model.head
            # For pose, num_keypoints is the target
            target_kpts = config.model.num_keypoints or config.model.num_classes
            logger.info(
                f"[{self.__class__.__name__}] Setting RTMO model.head.num_keypoints to {target_kpts}"
            )
            head.num_keypoints = target_kpts

            if hasattr(head, "head_module_cfg"):
                logger.info(
                    f"[{self.__class__.__name__}] Setting RTMO model.head.head_module_cfg.num_classes to 1"
                )
                head.head_module_cfg.num_classes = 1
