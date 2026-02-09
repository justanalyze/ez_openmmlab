from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, List
import tempfile

from loguru import logger
from mmengine.config import Config
from mmengine.runner import Runner

from ez_openmmlab.core.config_manager import ConfigManager, get_config_file
from ez_openmmlab.core.injectors import get_injectors
from ez_openmmlab.schemas.dataset import DatasetConfig
from ez_openmmlab.core.results import InferenceResult
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils.download import ensure_model_checkpoint
from ez_openmmlab.utils.path import get_unique_dir
from ez_openmmlab.utils.input import normalize_inputs
from ez_openmmlab.utils.context import switch_to_lib_root
from ez_openmmlab.utils.toml_config import (
    save_user_config,
    UserConfig,
)


class EZMMLab(ABC):
    """Abstract base class for all OpenMMLab libraries (Detection, Pose, etc.).

    Implements shared logic for configuration management and training workflows.
    """

    def __init__(
        self,
        model: Union[ModelName, str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
    ):
        """Initializes the library wrapper with a base model.

        Args:
            model: The name of the architecture (e.g., 'rtmdet_tiny') OR path to a config.toml.
            checkpoint_path: Path to a specific checkpoint (.pth or .pt).
            log_level: Global logging level. Default is 'INFO'.
        """
        # --- 1. Logging & Internal State ---
        self.log_level = log_level
        self._configure_logging(log_level)

        logger.info(f"Initializing {self.__class__.__name__} with model: '{model}'")

        # Ensure noisy warnings are suppressed when engine starts
        from ez_openmmlab import mute_warnings

        mute_warnings()

        # --- 2. Internal Managers ---
        self._config_manager = ConfigManager()

        # --- 3. Model Configuration & State ---
        self.checkpoint_path: Optional[Path] = None
        self.config_path: Optional[Path] = None
        self.model: Optional[str] = None
        self.num_classes: Optional[int] = None
        self.num_keypoints: Optional[int] = None
        self.metainfo: Optional[dict] = None

        self._cfg: Optional[Config] = None
        self._temp_config_file: Optional[Path] = None

        # --- 4. Initialization Sequence ---
        self._validate_inputs(model, checkpoint_path)
        self._resolve_resources(model, checkpoint_path)
        self._initialize_model_state(model)

    def _configure_logging(self, log_level: str) -> None:
        """Configures the global logger level."""
        try:
            logger.remove()
            import sys

            logger.add(sys.stderr, level=log_level)
        except Exception as e:
            logger.warning(f"Failed to set log level: {e}")

    def _validate_inputs(
        self,
        model: Union[ModelName, str, Path],
        checkpoint_path: Optional[Union[str, Path]],
    ) -> None:
        """Performs initial validation of provided arguments."""
        if isinstance(model, (Path, str)) and str(model).endswith(".toml"):
            if not checkpoint_path:
                raise ValueError(
                    "Checkpoint path is required when using a custom config.toml"
                )

    def _resolve_resources(
        self,
        model: Union[ModelName, str, Path],
        checkpoint_path: Optional[Union[str, Path]],
    ) -> None:
        """Resolves absolute paths for the model checkpoint and configuration."""
        # Resolve or download checkpoint
        if isinstance(model, (Path, str)) and str(model).endswith(".toml"):
            self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        else:
            self.checkpoint_path = ensure_model_checkpoint(model, checkpoint_path)

        # Resolve model configuration path
        if isinstance(model, (Path, str)) and str(model).endswith(".toml"):
            self._temp_config_file = self._config_manager.prepare_config_file(
                Path(model),
                Path(self.checkpoint_path) if self.checkpoint_path else None,
            )
            self.config_path = self._temp_config_file
        else:
            self.config_path = get_config_file(model)

    def _initialize_model_state(self, model: Union[ModelName, str, Path]) -> None:
        """Initializes the model name and loads metadata from the checkpoint."""
        # Determine model name
        if isinstance(model, ModelName):
            self.model = model.value
        elif isinstance(model, (Path, str)) and str(model).endswith(".toml"):
            meta = self._config_manager.load_metadata_from_checkpoint(
                Path(self.checkpoint_path) if self.checkpoint_path else Path(".")
            )
            self.model = meta.get("model_name")
        else:
            self.model = str(model)

        # Auto-load metadata if a checkpoint is present
        if self.checkpoint_path:
            meta = self._config_manager.load_metadata_from_checkpoint(
                Path(self.checkpoint_path)
            )
            self.num_classes = meta.get("num_classes")
            self.num_keypoints = meta.get("num_keypoints")
            self.metainfo = meta.get("metainfo")

    def __del__(self):
        """Cleanup temporary files."""
        if hasattr(self, "_temp_config_file"):
            self._config_manager.cleanup_temp_config(self._temp_config_file)

    @abstractmethod
    def predict(self, *args, **kwargs) -> List[InferenceResult]:
        """Abstract method for performing inference."""
        pass

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
    ) -> None:
        """Runs the end-to-end training pipeline.

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
        """
        target_log_level = log_level or self.log_level

        logger.info(f"Loading dataset configuration from: {dataset_config_path}")

        user_config = self._config_manager.build_user_config(
            model=self.model,
            dataset_config_path=dataset_config_path,
            checkpoint_path=self.checkpoint_path,
            epochs=epochs,
            batch_size=batch_size,
            device=device,
            work_dir=work_dir,
            learning_rate=learning_rate,
            amp=amp,
            num_workers=num_workers,
            enable_tensorboard=enable_tensorboard,
            log_level=target_log_level,
        )

        self._run_training_workflow(user_config)

    def _run_training_workflow(self, config: UserConfig) -> None:
        """Orchestrates the internal OpenMMLab setup and execution."""
        work_dir = Path(config.training.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)

        # Set the absolute path to the base python config for artifact tracking
        config.model.base_config_path = str(
            get_config_file(config.model.name).absolute()
        )

        save_user_config(config, work_dir / "user_config.toml")
        logger.info(f"User configuration saved to: {work_dir / 'user_config.toml'}")

        self._cfg = self._load_base_config(config.model.name)
        self._inject_user_configs(config)

        logger.info("Starting MMEngine Runner...")
        runner = Runner.from_cfg(self._cfg)
        runner.train()

    def _load_base_config(self, model: str) -> Config:
        config_path = get_config_file(model)

        with switch_to_lib_root(self.model) as lib_root:
            # Use relative path from lib_root
            rel_config_path = config_path.relative_to(lib_root)
            cfg = Config.fromfile(str(rel_config_path))

        return cfg

    def _inject_user_configs(self, config: UserConfig) -> None:
        """Applies configuration changes using the registered plugin injectors."""
        if not self._cfg:
            raise RuntimeError("Base config not loaded.")

        for injector in get_injectors(self.model):
            injector.apply(self._cfg, config)
