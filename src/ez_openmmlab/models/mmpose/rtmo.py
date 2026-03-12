from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from mmengine.config import Config
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.core.engines.mmpose import EZMMPose
from ez_openmmlab.core.schema.models import ModelName


class RTMO(EZMMPose):
    """Initializes a new RTMO engine.

    RTMO (Real-Time Multi-person One-stage) is a one-stage pose estimation model
    from MMPose. It detects people and their poses simultaneously in a single pass.

    Args:
        model: The model variant to use. Can be a member of :class:`ModelName`,
            a string name (e.g., 'rtmo_s'), or a path to a custom `config.toml`.
        checkpoint_path: Path to a custom model checkpoint (.pth). If None,
            the official pretrained weights will be automatically downloaded
            based on the `model` name.
        log_level: Global logging level for the engine. Defaults to "INFO".
        **kwargs: Additional configuration parameters passed to the base engine.
    """

    def _validate_model_name(self, name: str) -> None:
        supported = ["rtmo_s", "rtmo_m", "rtmo_l"]
        if name not in supported and not Path(name).exists():
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
        input_size: Tuple[int, int] = (640, 640),
        weight_decay: float = 0.05,
        evaluator_metric: Union[str, List[str]] = "CocoMetric",
        augments: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
        stage2_num_epochs: int = 20,
        **kwargs,
    ) -> None:
        """Runs a fresh RTMO training pipeline with architecture-specific parameters.

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
            augments: Dictionary of data augmentation parameters.
            dry_run: If True, only generates the final config file without starting training.
            stage2_num_epochs: Number of epochs for stage 2 training pipeline.
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
        """Resumes an RTMO training session.

        Args:
            checkpoint: Whether to resume. If True, automatically find the latest
                checkpoint in the source directory. If string, use as specific path.
            epochs: Optional override for total epochs.
            batch_size: Optional override for batch size.
            learning_rate: Optional override for learning rate.
            work_dir: Optional override for working directory.
            dry_run: If True, only generates the final config file without starting training.
            stage2_num_epochs: Optional override for stage 2 training pipeline epochs.
            **kwargs: Additional overrides.
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

    def _instantiate_inferencer(
        self, cfg: Config, device: str, **kwargs
    ) -> MMPoseInferencer:
        """Instantiates the RTMO inferencer."""
        return MMPoseInferencer(
            pose2d=cfg,
            pose2d_weights=str(self.checkpoint_path),
            device=device,
        )

