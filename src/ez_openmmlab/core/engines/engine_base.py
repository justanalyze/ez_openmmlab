from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional, Union

from loguru import logger
from mmengine.config import Config
from mmengine.runner import Runner

from ez_openmmlab.core.config_manager import ConfigManager, get_config_file
from ez_openmmlab.core.injectors import get_injectors
from ez_openmmlab.core.results import InferenceResult
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils.context import switch_to_lib_root
from ez_openmmlab.utils.download import ensure_model_checkpoint
from ez_openmmlab.utils.input import normalize_inputs
from ez_openmmlab.utils.path import get_unique_dir
from ez_openmmlab.utils.toml_config import (
    UserConfig,
    save_user_config,
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
        """Resolves absolute paths for resources and initializes model state.

        Args:
            model: ModelName enum or path to config.toml.
            checkpoint_path: Path to model weights (.pth).
        """
        # Case 1: Custom Configuration via TOML
        if isinstance(model, (Path, str)) and str(model).endswith(".toml"):
            config_toml = Path(model)
            self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None

            # 1.1 Load explicit metadata from TOML
            meta = self._config_manager.load_metadata_from_toml(config_toml)
            self.model = meta.get("model_name")
            self.num_classes = meta.get("num_classes")
            self.num_keypoints = meta.get("num_keypoints")
            self.metainfo = meta.get("metainfo")

            # 1.2 Generate temporary Python config
            self._temp_config_file = self._config_manager.prepare_config_file(
                config_toml, self.checkpoint_path
            )
            self.config_path = self._temp_config_file

        # Case 2: Standard Model Name
        else:
            self.model = model.value if isinstance(model, ModelName) else str(model)
            self.checkpoint_path = ensure_model_checkpoint(model, checkpoint_path)
            self.config_path = get_config_file(model)

    def __del__(self):
        """Cleanup temporary files."""
        if hasattr(self, "_temp_config_file"):
            self._config_manager.cleanup_temp_config(self._temp_config_file)

    def predict(
        self,
        image_path: Union[str, Path, list],
        *,
        device: str = "cuda",
        out_dir: Optional[str] = None,
        show: bool = False,
        **kwargs,
    ) -> List[InferenceResult]:
        """Universal prediction entry point (Template Method).

        Args:
            image_path: Path to a single image, a list of paths, or a directory.
            device: Computing device ('cuda', 'cpu').
            out_dir: Directory to save visualization images.
            show: Whether to pop up a window with the result.
            **kwargs: Library-specific inference arguments.
        """
        # 1. Lazy Initialization
        self._init_inferencer(device, **kwargs)

        # 2. Setup Resources
        actual_out_dir = str(get_unique_dir(out_dir)) if out_dir else ""
        inputs = normalize_inputs(image_path)

        if not hasattr(self, "_inferencer") or self._inferencer is None:
            raise RuntimeError("Inferencer failed to initialize.")

        # 3. Delegate execution to child
        raw_results = self._run_inference(inputs, actual_out_dir, show, **kwargs)

        # 4. Format results
        if not hasattr(self, "_formatter") or self._formatter is None:
            raise RuntimeError("Result formatter not initialized.")

        return self._formatter.map_results(raw_results, inputs, self._get_class_names())

    @abstractmethod
    def _init_inferencer(self, device: str, **kwargs) -> None:
        """Library-specific inferencer initialization."""
        pass

    @abstractmethod
    def _run_inference(
        self, inputs: list, out_dir: str, show: bool, **kwargs
    ) -> Union[dict, list]:
        """Library-specific inference execution."""
        pass

    def _get_class_names(self) -> dict:
        """Retrieves class names from local metainfo or inferencer.

        Returns:
            A dictionary mapping class IDs to names.
        """
        # 1. Check local metainfo (auto-loaded from config near checkpoint)
        if self.metainfo and "classes" in self.metainfo:
            return {i: name for i, name in enumerate(self.metainfo["classes"])}

        # 2. Check inferencer model metadata (contains model's original training classes)
        if hasattr(self, "_inferencer") and self._inferencer:
            # Handle MMDet/MMPose internal model metadata
            model = getattr(self._inferencer, "model", None)
            if model:
                meta = getattr(model, "dataset_meta", {})
                if "classes" in meta:
                    return {i: name for i, name in enumerate(meta["classes"])}

        # 3. No fallback
        return {}

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
