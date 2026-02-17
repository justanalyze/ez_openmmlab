from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from mmengine.config import Config
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.core.engines.mmpose import EZMMPose
from ez_openmmlab.schemas.model import RTMO_CONFIGS, ModelName


class RTMO(EZMMPose):
    """RTMO implementation for one-stage, multi-person 2D pose estimation."""

    def __init__(
        self,
        model: Union[ModelName, str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
        **kwargs,
    ):
        """Initializes a new RTMO engine.

        RTMO (Towards High-Performance One-Stage Real-Time Multi-Person Pose Estimation)
        is a bottom-up model, meaning it detects people and their poses in a single
        pass without requiring a separate object detector.

        Args:
            model: The model variant to use. Supported: 'rtmo_s', 'rtmo_m', 'rtmo_l'.
                Can be a :class:`ModelName`, string, or path to a `config.toml`.
            checkpoint_path: Path to a custom model checkpoint (.pth). If None,
                the official weights will be downloaded automatically.
            log_level: Global logging level for the engine. Defaults to "INFO".
            **kwargs: Additional configuration parameters passed to the base engine.
        """
        self._validate_model(model)
        super().__init__(model, checkpoint_path, log_level, **kwargs)

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

    def _get_architecture_params(self, **kwargs) -> Dict[str, Any]:
        """Extracts RTMO specific parameters."""
        return {
            "input_size": kwargs.get("input_size"),
        }

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
        # RTMO Specific
        input_size: Tuple[int, int] = (640, 640),
        weight_decay: float = 0.05,
        evaluator_metric: Union[str, List[str]] = "CocoMetric",
        resume: Union[bool, str] = False,
        **kwargs,
    ) -> None:
        """Runs the RTMO training pipeline with architecture-specific parameters.

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
            input_size: Target resolution (width, height). Defaults to (640, 640).
            weight_decay: Optimizer weight decay. Defaults to 0.05.
            evaluator_metric: Metric(s) for validation. Defaults to "CocoMetric".
            resume: Whether to resume training.
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
            resume=resume,
            **kwargs,
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