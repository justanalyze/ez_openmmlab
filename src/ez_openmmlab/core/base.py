from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union, List
import tempfile

from loguru import logger
from mmengine.config import Config
from mmengine.runner import Runner

from ez_openmmlab.core.config_loader import get_config_file
from ez_openmmlab.core.handlers import DataloaderHandler, RuntimeHandler
from ez_openmmlab.schemas.dataset import DatasetConfig
from ez_openmmlab.core.results import InferenceResult
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils.download import ensure_model_checkpoint
from ez_openmmlab.utils.path import get_unique_dir
from ez_openmmlab.utils.toml_config import (
    DataSection,
    ModelSection,
    TrainingSection,
    UserConfig,
    save_user_config,
    load_user_config,
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
        logger.info(
            f"Initializing {self.__class__.__name__} with base model: '{model}'"
        )
        # Ensure noisy warnings are suppressed when engine starts
        from ez_openmmlab import mute_warnings

        mute_warnings()

        self.log_level: str = log_level
        self.num_classes: Optional[int] = None
        self.num_keypoints: Optional[int] = None
        self.metainfo: Optional[dict] = None
        self._cfg: Optional[Config] = None
        self._temp_config_file: Optional[Path] = None

        # Resolve or download checkpoint (might be None if training from scratch, but required for custom inference)
        # Note: ensure_model_checkpoint might need adjustment if model is a path, 
        # but for now we assume if it's a path, the user MUST provide a checkpoint_path.
        if isinstance(model, (Path, str)) and str(model).endswith(".toml"):
             self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else None
        else:
             self.checkpoint_path = ensure_model_checkpoint(model, checkpoint_path)

        # Resolve model configuration
        # This will set self.config_path to either a standard config or a temp config
        self.config_path = self._resolve_model_config(model)
        
        # If we loaded from a config.toml, we should set the internal model attribute to the base model
        # so other parts of the system (like training logs) make sense.
        # _resolve_model_config sets self.model as a side effect if it parses a toml.
        if not hasattr(self, 'model'):
             self.model = model.value if isinstance(model, ModelName) else str(model)

        # If a custom checkpoint is used, try to auto-load metadata from accompanying config
        if self.checkpoint_path:
            self._auto_load_metadata(Path(self.checkpoint_path))

        # Configure loguru level
        try:
            logger.remove()
            import sys

            logger.add(sys.stderr, level=log_level)
        except Exception as e:
            logger.warning(f"Failed to set log level: {e}")

    def _resolve_model_config(self, model_input: Union[str, Path, ModelName]) -> Path:
        """Resolves the model configuration path.

        If input is a ModelName, returns the standard config path.
        If input is a config.toml path, generates a temporary python config.
        """
        # Case 1: Custom config.toml
        toml_path = Path(model_input)
        if toml_path.suffix == ".toml":
            if not self.checkpoint_path:
                raise ValueError("Checkpoint path is required when using a custom config.toml")
            
            if not toml_path.exists():
                raise FileNotFoundError(f"Custom config file not found: {toml_path}")

            logger.info(f"Loading custom configuration from: {toml_path}")
            user_cfg = load_user_config(toml_path)
            
            # Set internal state from config
            self.model = user_cfg.model.name.value
            self.num_classes = user_cfg.model.num_classes
            self.num_keypoints = user_cfg.model.num_keypoints
            if user_cfg.data.metainfo:
                self.metainfo = user_cfg.data.metainfo

            # Resolve base config
            base_config_path = get_config_file(self.model)
            
            # Create temporary config inheriting from base
            # We use the absolute path of the base config
            base_config_str = str(base_config_path.absolute())
            
            # Create content for temp config
            # We override essential parameters. Detailed dataset/pipeline overrides happen in handlers during training,
            # but for inference or initial setup, we might need basic model structure.
            # Ideally, the engine's _init_inferencer will load this config.
            content = f'_base_ = ["{base_config_str}"]\n'
            
            # Create temp file
            # We keep it until the object is destroyed or explicitly cleaned up.
            # Using NamedTemporaryFile with delete=False so we can pass the path around.
            temp_file = tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False)
            temp_file.write(temp_file.name if isinstance(temp_file.name, str) else str(temp_file.name)) # Dummy use to avoid lint issue if any, but NamedTemporaryFile returns name as attribute
            temp_file.write(content)
            temp_file.close()
            
            self._temp_config_file = Path(temp_file.name)
            logger.debug(f"Created temporary config file: {self._temp_config_file}")
            return self._temp_config_file

        # Case 2: Standard ModelName (or string equivalent)
        # ensure_model_checkpoint already handled weights, we just need the config
        return get_config_file(model_input)

    def __del__(self):
        """Cleanup temporary files."""
        if hasattr(self, '_temp_config_file') and self._temp_config_file and self._temp_config_file.exists():
            try:
                self._temp_config_file.unlink()
                logger.debug(f"Removed temporary config file: {self._temp_config_file}")
            except Exception as e:
                logger.warning(f"Failed to remove temp config file: {e}")

    def _auto_load_metadata(self, checkpoint_path: Path) -> None:
        """Attempts to find and load training metadata (num_classes, keypoints) from nearby config files."""
        search_dirs = [checkpoint_path.parent, checkpoint_path.parent.parent]
        config_files = ["user_config.toml", "dataset.toml"]

        for d in search_dirs:
            for f in config_files:
                path = d / f
                if path.exists():
                    try:
                        if f == "user_config.toml":
                            from ez_openmmlab.utils.toml_config import load_user_config
                            user_cfg = load_user_config(path)
                            self.num_classes = user_cfg.model.num_classes
                            self.num_keypoints = user_cfg.model.num_keypoints
                            self.metainfo = user_cfg.data.metainfo
                            logger.info(f"Auto-loaded metadata from: {path}")
                        else:
                            from ez_openmmlab.schemas.dataset import DatasetConfig
                            ds_cfg = DatasetConfig.from_toml(path)
                            self.num_classes = len(ds_cfg.classes) if ds_cfg.classes else None
                            if ds_cfg.metainfo:
                                self.metainfo = ds_cfg.metainfo
                                if "keypoint_info" in ds_cfg.metainfo:
                                    self.num_keypoints = len(ds_cfg.metainfo["keypoint_info"])
                            logger.info(f"Auto-loaded metadata from: {path}")
                        return
                    except Exception as e:
                        logger.warning(f"Failed to auto-load metadata from {path}: {e}")

    @abstractmethod
    def predict(self, *args, **kwargs) -> List[InferenceResult]:
        """Abstract method for performing inference."""
        pass

    def _resolve_out_dir(self, out_dir: Optional[str]) -> str:
        """Returns a unique output directory if one is provided."""
        return str(get_unique_dir(out_dir)) if out_dir else ""

    def _normalize_inputs(
        self, inputs: Union[str, Path, List[Union[str, Path]]]
    ) -> Union[str, List[str]]:
        """Normalizes input paths (single, list, or directory) into a format engines accept.

        Args:
            inputs: A single path, a list of paths, or a directory path.

        Returns:
            A list of image paths (strings) if the input was a list or directory,
            or a single string path if the input was a single file.
        """
        # Case 1: List of paths -> Convert all to strings
        if isinstance(inputs, list):
            return [str(p) for p in inputs]

        path_obj = Path(inputs)

        # Case 2: Directory -> Glob all images
        if path_obj.is_dir():
            extensions = {"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"}
            images = []
            for ext in extensions:
                # Case insensitive globbing is harder in pure pathlib, so we iterate
                images.extend([str(p) for p in path_obj.glob(ext)])
                images.extend([str(p) for p in path_obj.glob(ext.upper())])

            if not images:
                logger.warning(f"No images found in directory: {inputs}")
            return sorted(list(set(images)))

        # Case 3: Single file path -> Return as string
        return str(path_obj)

    def switch_to_lib_root(self):
        """Context manager to temporarily switch to the appropriate library root.

        This is necessary because OpenMMLab configs use relative paths
        that expect to be resolved from the library root (libs/mmdet or libs/mmpose).
        """
        import os
        from contextlib import contextmanager

        @contextmanager
        def _switch():
            if "rtmpose" in self.model or "rtmo" in self.model:
                lib_root = Path.cwd() / "libs" / "mmpose"
            else:
                lib_root = Path.cwd() / "libs" / "mmdetection"

            old_cwd = os.getcwd()
            try:
                os.chdir(lib_root)
                yield lib_root
            finally:
                os.chdir(old_cwd)

        return _switch()

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
        dataset_cfg = DatasetConfig.from_toml(Path(dataset_config_path))

        self.classes = dataset_cfg.classes
        self.num_classes: int = len(dataset_cfg.classes) if dataset_cfg.classes else 80

        # Extract num_keypoints from metainfo if available (for pose models)
        num_keypoints = None
        if dataset_cfg.metainfo and "keypoint_info" in dataset_cfg.metainfo:
            num_keypoints = len(dataset_cfg.metainfo["keypoint_info"])

        user_config = UserConfig(
            model=ModelSection(
                name=self.model,
                num_classes=self.num_classes,
                num_keypoints=num_keypoints,
                load_from=str(self.checkpoint_path),
            ),
            data=DataSection(
                root=str(dataset_cfg.data_root),
                train_ann=dataset_cfg.train.ann_file,
                train_img=dataset_cfg.train.img_dir,
                val_ann=dataset_cfg.val.ann_file,
                val_img=dataset_cfg.val.img_dir,
                test_ann=dataset_cfg.test.ann_file if dataset_cfg.test else None,
                test_img=dataset_cfg.test.img_dir if dataset_cfg.test else None,
                classes=self.classes,
                metainfo=dataset_cfg.metainfo,
            ),
            training=TrainingSection(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                device=device,
                work_dir=work_dir,
                log_level=target_log_level,
                amp=amp,
                num_workers=num_workers,
                enable_tensorboard=enable_tensorboard,
            ),
        )

        self._run_training_workflow(user_config)

    def _run_training_workflow(self, config: UserConfig) -> None:
        """Orchestrates the internal OpenMMLab setup and execution."""
        work_dir = Path(config.training.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        save_user_config(config, work_dir / "user_config.toml")
        logger.info(f"User configuration saved to: {work_dir / 'user_config.toml'}")

        self._cfg = self._load_base_config(config.model.name)
        self._apply_common_overrides(config)

        logger.info(
            f"Configuring architecture specifics for {self.__class__.__name__}..."
        )
        self._configure_model_specifics(config)

        logger.info("Starting MMEngine Runner...")
        runner = Runner.from_cfg(self._cfg)
        runner.train()

    def _load_base_config(self, model: str) -> Config:
        config_path = get_config_file(model)

        with self.switch_to_lib_root() as lib_root:
            # Use relative path from lib_root
            rel_config_path = config_path.relative_to(lib_root)
            cfg = Config.fromfile(str(rel_config_path))

        return cfg

    def _apply_common_overrides(self, config: UserConfig) -> None:
        """Applies configuration changes common to all architectures."""
        if not self._cfg:
            raise RuntimeError("Base config not loaded.")

        DataloaderHandler().apply(self._cfg, config)
        RuntimeHandler().apply(self._cfg, config)

    @abstractmethod
    def _configure_model_specifics(self, config: UserConfig) -> None:
        """Architecture-specific overrides."""
        pass
