from pathlib import Path
from typing import List, Optional, Tuple, Union

from mmengine.config import Config
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.core.config_manager import get_config_file
from ez_openmmlab.core.engines.mmpose import EZMMPose
from ez_openmmlab.schemas.model import RTM_POSE_CONFIGS, ModelName
from ez_openmmlab.utils.download import ensure_model_checkpoint


class RTMPose(EZMMPose):
    """RTMPose implementation for high-performance 2D keypoint estimation."""

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
        """Initializes a new RTMPose engine.

        RTMPose is a top-down pose estimation model family. Unlike bottom-up models,
        it requires a bounding box detector (like RTMDet) to first identify persons
        before estimating their keypoints.

        Args:
            model: The pose model variant to use. Supported: 'rtmpose_tiny', 'rtmpose_s',
                'rtmpose_m', 'rtmpose_l'. Can be :class:`ModelName`, string, or TOML path.
            checkpoint_path: Path to custom pose weights (.pth). If None, official
                weights are downloaded automatically.
            log_level: Global logging level for the engine. Defaults to "INFO".
            det_model: The detector model to use for person detection.
                Defaults to "rtmdet_tiny".
            det_weights: Path to custom detector weights. If None, official
                weights for `det_model` are downloaded.
            det_cat_ids: Category IDs in the detector output to treat as "person".
                Defaults to [0] (COCO Person).
            **kwargs: Additional configuration parameters passed to the base engine.
        """
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

                det_config = self.det_model

                # Only resolve to absolute path if it looks like a file path and exists

                # This preserves registry keys (e.g. 'yolox_tiny_8x8...') while handling local configs

                if isinstance(det_config, (str, Path)) and (

                    Path(det_config).exists() or "/" in str(det_config)

                ):

                    det_config = str(Path(det_config).absolute())

    

                det_weights = str(self.det_weights) if self.det_weights else None

    

            return det_config, det_weights, self.det_cat_ids

    

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

            # RTMPose Specific

            input_size: Tuple[int, int] = (192, 256),

            simcc_sigma: Optional[Tuple[float, float]] = None,

            feature_map_size: Optional[Tuple[int, int]] = None,

            weight_decay: float = 0.05,

            evaluator_metric: Union[str, List[str]] = "CocoMetric",

            **kwargs,

        ) -> None:

            """Runs the RTMPose training pipeline with architecture-specific parameters.

    

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

                input_size=input_size,

                simcc_sigma=simcc_sigma,

                feature_map_size=feature_map_size,

                weight_decay=weight_decay,

                evaluator_metric=evaluator_metric,

                **kwargs,

            )

    