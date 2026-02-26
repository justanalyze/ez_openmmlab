from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from mmengine.config import Config
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.core.engines.mmpose import EZMMPose
from ez_openmmlab.core.inference.results import InferenceResult
from ez_openmmlab.core.schema.models import ModelName


class RTMPose(EZMMPose):
    """Top-down RTMPose architecture."""

    def _validate_model_name(self, name: str) -> None:
        supported = ["rtmpose_tiny", "rtmpose_s", "rtmpose_m", "rtmpose_l"]
        if name not in supported and not Path(name).exists():
            raise ValueError(
                f"Invalid model variant '{name}' for RTMPose. "
                f"Supported variants: {supported}, or a path to a custom config.toml"
            )

    def _get_architecture_params(self, **kwargs) -> Dict[str, Any]:
        """Extracts RTMPose specific parameters."""
        return {
            "input_size": kwargs.get("input_size"),
            "simcc_sigma": kwargs.get("simcc_sigma"),
            "feature_map_size": kwargs.get("feature_map_size"),
        }

    def _instantiate_inferencer(
        self, cfg: Config, device: str, **kwargs
    ) -> MMPoseInferencer:
        """Initializes the top-down RTMPose inferencer with a detector."""
        det_config, det_weights, det_cat_ids = self._resolve_detector_params(**kwargs)

        return MMPoseInferencer(
            pose2d=cfg,
            pose2d_weights=str(self.checkpoint_path),
            device=device,
            det_model=det_config,
            det_weights=det_weights,
            det_cat_ids=det_cat_ids,
        )

    def _resolve_detector_params(
        self, **kwargs
    ) -> Tuple[Optional[str], Optional[str], List[int]]:
        """Resolves detector configuration for top-down inference."""
        det_model = kwargs.get("det_model")
        det_weights = kwargs.get("det_weights")
        det_cat_ids = kwargs.get("det_cat_ids", [0])  # Default to person class

        # If no custom detector provided, use default RTMDet-Tiny
        det_config = None
        det_weights_path = None

        if det_model is None:
            from ez_openmmlab.core.config_manager import get_config_file

            det_config = str(get_config_file(ModelName.RTM_DET_TINY))
        else:
            # Handle ModelName enum
            if isinstance(det_model, ModelName):
                from ez_openmmlab.core.config_manager import get_config_file

                det_config = str(get_config_file(det_model))
            else:
                det_config = str(det_model)

            # Ensure absolute path if local file
            if isinstance(det_config, (str, Path)) and (
                Path(det_config).exists() or "/" in str(det_config)
            ):
                det_config = str(Path(det_config).absolute())

            det_weights_path = str(det_weights) if det_weights else None

        return det_config, det_weights_path, det_cat_ids

    def predict(
        self,
        image_path: Union[str, Path, list],
        *,
        device: str = "cuda",
        out_dir: Optional[str] = None,
        show: bool = False,
        # Top-down specific overrides
        det_model: Optional[Union[ModelName, str, Path]] = None,
        det_weights: Optional[Union[str, Path]] = None,
        det_cat_ids: Optional[List[int]] = None,
        **kwargs,
    ) -> List[InferenceResult]:
        """Universal prediction entry point for top-down pose estimation.

        Args:
            image_path: Path to a single image, a list of paths, or a directory.
            device: Computing device ('cuda', 'cpu').
            out_dir: Directory to save visualization images.
            show: Whether to pop up a window with the result.
            det_model: Optional override for the detector model. Defaults to 'rtmdet_tiny'.
            det_weights: Optional override for detector weights.
            det_cat_ids: Optional override for detector category IDs. Defaults to [0].
            **kwargs: Additional inference arguments.
        """
        # Warn if user tries to override detector after inferencer is already initialized
        if self._inferencer is not None and (det_model or det_weights or det_cat_ids):
            logger.warning(
                "Inferencer is already initialized. Overriding detector parameters "
                "in predict() will have no effect unless you re-instantiate the RTMPose object."
            )

        return super().predict(
            image_path,
            device=device,
            out_dir=out_dir,
            show=show,
            det_model=det_model,
            det_weights=det_weights,
            det_cat_ids=det_cat_ids,
            **kwargs,
        )

    def train(
        self,
        dataset_config_path: Union[str, Path],
        epochs: int = 100,
        batch_size: int = 8,
        device: str = "cuda",
        work_dir: str = "./runs/train",
        learning_rate: float = 0.001,
        amp: bool = True,
        num_workers: int = 4,
        enable_tensorboard: bool = False,
        log_level: Optional[str] = None,
        weight_decay: float = 0.05,
        evaluator_metric: Union[str, List[str]] = "CocoMetric",
        augments: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
        stage2_num_epochs: int = 20,
        # RTMPose Specific
        input_size: Tuple[int, int] = (192, 256),
        simcc_sigma: Optional[Tuple[float, float]] = None,
        feature_map_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ) -> None:
        """Runs a fresh RTMPose training pipeline with architecture-specific parameters.

        Args:
            dataset_config_path: Path to the dataset.toml definition.
            epochs: Number of training epochs.
            batch_size: Total batch size for training.
            device: Training hardware ('cuda', 'cpu', 'mps').
            work_dir: Directory for logs and checkpoints.
            learning_rate: Initial learning rate.
            amp: Enable Automatic Mixed Precision.
            num_workers: Number of data loading workers.
            enable_tensorboard: Enable TensorBoard visualization.
            log_level: Override for internal framework logging.
            input_size: Target resolution (width, height). Defaults to (192, 256).
            simcc_sigma: Sigma for SimCC heatmap generation. If None, scales with resolution.
            feature_map_size: Custom feature map size. If None, derived as input_size // 32.
            weight_decay: Optimizer weight decay. Defaults to 0.05.
            evaluator_metric: Metric(s) for validation. Defaults to "CocoMetric".
            augments: Dictionary of data augmentation parameters.
            dry_run: If True, only generates the final config file without starting training.
            stage2_num_epochs: Number of epochs for stage 2 training pipeline.
            det_model: Optional detector model name for evaluation.
            det_weights: Optional path to detector weights.
            det_cat_ids: Optional detector category IDs.
            **kwargs: Additional parameters.
        """
        super().train(
            dataset_config_path=dataset_config_path,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            work_dir=work_dir,
            learning_rate=learning_rate,
            amp=amp,
            num_workers=num_workers,
            enable_tensorboard=enable_tensorboard,
            log_level=log_level,
            weight_decay=weight_decay,
            evaluator_metric=evaluator_metric,
            input_size=input_size,
            simcc_sigma=simcc_sigma,
            feature_map_size=feature_map_size,
            augments=augments,
            dry_run=dry_run,
            stage2_num_epochs=stage2_num_epochs,
            **kwargs,
        )

    def resume(
        self,
        checkpoint: Union[bool, str] = True,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        work_dir: Optional[str] = None,
        dry_run: bool = False,
        stage2_num_epochs: Optional[int] = None,
        **kwargs,
    ) -> None:
        """Resumes an RTMPose training session.

        Args:
            checkpoint: Whether to resume. If True, automatically find the latest
                checkpoint in the source directory. If string, use as specific path.
            epochs: Optional override for total epochs.
            batch_size: Optional override for batch size.
            learning_rate: Optional override for learning rate.
            work_dir: Optional override for working directory.
            dry_run: If True, only generates the final config file without starting training.
            stage2_num_epochs: Optional override for stage 2 training pipeline epochs.
            **kwargs: Additional overrides (e.g. det_model, etc.).
        """
        super().resume(
            checkpoint=checkpoint,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            work_dir=work_dir,
            dry_run=dry_run,
            stage2_num_epochs=stage2_num_epochs,
            **kwargs,
        )

