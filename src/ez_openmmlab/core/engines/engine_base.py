from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from mmengine.config import Config
from mmengine.runner import Runner

from ez_openmmlab.core.config_manager import ConfigManager, get_config_file
from ez_openmmlab.core.datasets import DynamicDatasetRegistry
from ez_openmmlab.core.inference.results import InferenceResult
from ez_openmmlab.core.surgery import get_surgeries
from ez_openmmlab.core.schema.models import ModelName
from ez_openmmlab.core.utils.context import switch_to_lib_root
from ez_openmmlab.core.utils.download import ensure_model_checkpoint
from ez_openmmlab.core.utils.input import normalize_inputs
from ez_openmmlab.core.utils.path import get_unique_dir
from ez_openmmlab.core.schema.config import (
    DataSection,
    ModelSection,
    TrainingSection,
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
        **kwargs,
    ):
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
        self.num_classes: Optional[int] = kwargs.get("num_classes")
        self.num_keypoints: Optional[int] = kwargs.get("num_keypoints")
        self.metainfo: Optional[dict] = kwargs.get("metainfo")
        self.architecture_params: Dict[str, Any] = {}

        self._cfg: Optional[Config] = None
        self._temp_config_file: Optional[Path] = None
        self._source_dir: Optional[Path] = None  # Directory of the source config
        self._source_toml: Optional[Path] = None  # Path to the source TOML file
        self._using_custom_weights: bool = checkpoint_path is not None

        # --- 4. Initialization Sequence ---
        self._resolve_resources(model, checkpoint_path)
        self._validate_inputs(model, self.checkpoint_path)

    def _validate_augments(self, augments: Optional[Dict[str, Any]]) -> None:
        """Strictly validates augmentation keys against the registry."""
        if not augments:
            return

        from ez_openmmlab.core.surgery.pipeline_patchers import (
            PipelineTransformPatcherRegistry,
        )

        family = self._get_library_family()
        supported = PipelineTransformPatcherRegistry.get_supported_augments(family)

        for key in augments:
            if key not in supported:
                raise ValueError(
                    f"Unsupported augmentation key '{key}' for {self.__class__.__name__}. "
                    f"Available augmentations for this model: {supported}"
                )

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
        is_toml = isinstance(model, (Path, str)) and str(model).endswith(".toml")

        # 1. Check for missing checkpoint in custom config context
        if is_toml and not checkpoint_path:
            # We allow missing checkpoint if we are LOADING for training resume,
            # but we warn the user that inference will fail.
            logger.warning(
                f"No checkpoint resolved for custom config: {model}. "
                "Inference will fail, but you can still call .resume() "
                "to continue training using MMEngine's auto-resume."
            )

        # 2. Enforce explicit configuration for custom weights to prevent head size mismatches
        # This applies when a user provides weights manually (Case 2 in resolve_resources)
        if checkpoint_path and self._using_custom_weights and not is_toml:
            if self.num_classes is None and self.num_keypoints is None:
                raise ValueError(
                    f"You provided custom weights ({checkpoint_path}) but no custom configuration. "
                    "To load a custom trained model, please provide its 'config.toml' as the 'model' argument. "
                    "Alternatively, specify 'num_classes' explicitly if using a standard model name."
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
            self._source_toml = config_toml.absolute()
            self._source_dir = config_toml.parent.absolute()

            # Smart Resolution: Attempt to find checkpoint in TOML directory if not provided
            if checkpoint_path:
                self.checkpoint_path = Path(checkpoint_path)
            else:
                self.checkpoint_path = self._try_resolve_checkpoint(self._source_dir)

            # 1.1 Load explicit metadata from TOML
            meta = self._config_manager.load_metadata_from_toml(config_toml)
            self.model = meta.get("model_name")
            self.num_classes = meta.get("num_classes")
            self.num_keypoints = meta.get("num_keypoints")
            self.metainfo = meta.get("metainfo")
            self.architecture_params = meta.get("architecture_params", {})

            # --- Validation: TOML must contain required metadata ---
            if self.num_classes is None:
                raise ValueError(
                    f"Metadata 'num_classes' is missing in '{config_toml}'. "
                    "When providing a custom config.toml, you must explicitly specify the number of classes."
                )

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

    def _try_resolve_checkpoint(self, directory: Path) -> Optional[Path]:
        """Smartly attempts to find a checkpoint in the given directory.

        Priority:
        1. best_*.pth
        2. Content of 'last_checkpoint' file
        """
        # 1. Search for 'best' checkpoint
        best_ckpts = list(directory.glob("best_*.pth"))
        if best_ckpts:
            # Pick the most recently modified 'best' checkpoint
            resolved = max(best_ckpts, key=lambda p: p.stat().st_mtime)
            logger.info(f"Smart-resolved 'best' checkpoint: {resolved.name}")
            return resolved

        # 2. Fallback to 'last_checkpoint' tracker
        last_ckpt_tracker = directory / "last_checkpoint"
        if last_ckpt_tracker.exists():
            try:
                ckpt_name = last_ckpt_tracker.read_text().strip()
                resolved = directory / ckpt_name
                if resolved.exists():
                    logger.info(
                        f"Smart-resolved last checkpoint from tracker: {resolved.name}"
                    )
                    return resolved
            except Exception as e:
                logger.warning(f"Failed to read 'last_checkpoint' tracker: {e}")

        return None

    def __del__(self):
        """Cleanup temporary files."""
        if hasattr(self, "_temp_config_file"):
            self._config_manager.cleanup_temp_config(self._temp_config_file)

    def _load_and_patch_config(self, **kwargs) -> Config:
        """Loads the model config and applies all registered plugin surgeries."""
        with switch_to_lib_root(self.model):
            cfg = Config.fromfile(str(self.config_path))

            # Trigger patching if custom metadata or architecture params are provided
            dummy_user_cfg = self._get_dummy_user_config(**kwargs)
            for surgery in get_surgeries(self.model):
                surgery.apply(cfg, dummy_user_cfg)

            return cfg

    def _get_dummy_user_config(self, **kwargs) -> UserConfig:
        """Creates a dummy UserConfig to satisfy the injector interface.

        This ensures that architecture-specific parameters passed during predict()
        or loaded from config.toml are correctly picked up by injectors.
        """
        model_params = {
            "name": self.model,
            "num_classes": self.num_classes,
            "num_keypoints": self.num_keypoints,
        }
        # 1. Use stored architecture_params (from config.toml)
        model_params.update(self.architecture_params)

        # 2. Inject architecture-specific parameters passed to predict()
        model_params.update(kwargs)

        return UserConfig(
            model=ModelSection(**model_params),
            training=TrainingSection(
                num_workers=0,
                learning_rate=0.001,
                weight_decay=None,
                evaluator_metric=None,
            ),
            data=DataSection(root=""),
        )

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
        if not self.checkpoint_path:
            raise ValueError(
                "No checkpoint found or provided. Inference requires a valid weight file (.pth)."
            )

        # 1. Extract and normalize architecture parameters
        # Use stored architecture_params as defaults for kwargs
        merged_kwargs = {**self.architecture_params, **kwargs}
        architecture_params = self._get_architecture_params(**merged_kwargs)

        # 2. Lazy Initialization with merged params
        self._init_inferencer(device, **{**architecture_params, **kwargs})

        # 3. Setup Resources
        actual_out_dir = str(get_unique_dir(out_dir)) if out_dir else ""
        inputs = normalize_inputs(image_path)

        if not hasattr(self, "_inferencer") or self._inferencer is None:
            raise RuntimeError("Inferencer failed to initialize.")

        # 4. Delegate execution to child
        raw_results = self._run_inference(inputs, actual_out_dir, show, **kwargs)

        # 5. Format results
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

    @abstractmethod
    def _get_library_family(self) -> str:
        """Returns the library family ('mmdet' or 'mmpose') for this engine."""
        pass

    @abstractmethod
    def _get_architecture_params(self, **kwargs) -> Dict[str, Any]:
        """Extracts architecture-specific hyperparameters from kwargs."""
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
        weight_decay: float = 0.05,
        evaluator_metric: Union[str, List[str]] = "CocoMetric",
        augments: Optional[Dict[str, Any]] = None,
        dry_run: bool = False,
        **kwargs,
    ) -> None:
        """Runs a fresh end-to-end training pipeline.

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
            weight_decay: Optimizer weight decay.
            evaluator_metric: Metric(s) for validation.
            augments: Dictionary of data augmentation parameters.
            dry_run: If True, only generates the final config file without starting training.
            **kwargs: Additional architecture-specific parameters.
        """
        logger.info(f"Assembling fresh training config for: {dataset_config_path}")
        self._validate_augments(augments)
        architecture_params = self._get_architecture_params(**kwargs)

        # Extract individual augmentation values for create_fresh_config
        aug_dict = augments or {}
        scale_factor = aug_dict.get("scale_factor")
        rotate_factor = aug_dict.get("rotate_factor")
        random_flip_prob = aug_dict.get("random_flip_prob")

        user_config = self._config_manager.create_fresh_config(
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
            log_level=log_level or self.log_level,
            weight_decay=weight_decay,
            evaluator_metric=evaluator_metric,
            architecture_params=architecture_params,
            scale_factor=scale_factor,
            rotate_factor=rotate_factor,
            random_flip_prob=random_flip_prob,
            **kwargs,
        )

        self._run_training_workflow(user_config, dry_run=dry_run)

    def resume(
        self,
        checkpoint: Union[bool, str] = True,
        epochs: Optional[int] = None,
        batch_size: Optional[int] = None,
        learning_rate: Optional[float] = None,
        work_dir: Optional[str] = None,
        dry_run: bool = False,
        **kwargs,
    ) -> None:
        """Resumes an unfinished training session from a source directory.

        Args:
            checkpoint: Whether to resume. If True, automatically find the latest
                checkpoint in the source directory. If string, use as specific path.
            epochs: Optional override for the total number of epochs.
            batch_size: Optional override for training batch size.
            learning_rate: Optional override for the learning rate.
            work_dir: Optional override for the working directory. If None,
                it defaults to the directory containing the source configuration.
            dry_run: If True, only generates the final config file without starting training.
            **kwargs: Additional training parameter overrides.
        """
        if not self._source_toml:
            raise ValueError(
                "Training resumption requires the model to be initialized with "
                "the 'user_config.toml' from the previous run."
            )

        # Resolve effective directory context
        effective_work_dir = work_dir or str(self._source_dir)
        logger.info(f"Resuming training in context: {effective_work_dir}")

        # Recover and patch context
        user_config = self._config_manager.recover_config_from_toml(
            toml_path=self._source_toml,
            resume=checkpoint,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            work_dir=effective_work_dir,
            **kwargs,
        )

        self._run_training_workflow(user_config, dry_run=dry_run)

    def _run_training_workflow(self, config: UserConfig, dry_run: bool = False) -> None:
        """Orchestrates the internal OpenMMLab setup and execution."""
        # 1. Register Dataset
        config.data.registered_class_name = DynamicDatasetRegistry.register_dataset(
            config, self._get_library_family()
        )

        # 2. Setup Artifacts
        work_dir = Path(config.training.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        config.model.base_config_path = str(
            get_config_file(config.model.name).absolute()
        )

        save_user_config(config, work_dir / "user_config.toml")
        logger.info(f"User configuration saved to: {work_dir / 'user_config.toml'}")

        # 3. Prepare Final Configuration
        self._cfg = self._load_base_config(config.model.name)
        self._inject_user_configs(config)

        final_config_path = (
            work_dir / f"{config.model.name.value}_{config.data.dataset_name}.py"
        )
        self._config_manager.dump_config(self._cfg, final_config_path)

        if dry_run:
            logger.info(f"Dry run enabled. Config generated at: {final_config_path}")
            return

        # 4. Execute Training
        self._run_training(final_config_path, config.training.log_level)

        # 5. Synchronize State
        self._sync_state_after_training(work_dir, final_config_path)

    def _sync_state_after_training(self, work_dir: Path, config_path: Path) -> None:
        """Synchronizes engine state with newly trained weights and persistent configs."""
        logger.info("Synchronizing model state with newly trained weights...")

        # Resolve weights
        new_checkpoint = self._try_resolve_checkpoint(work_dir)
        if new_checkpoint:
            self.checkpoint_path = new_checkpoint
            self._using_custom_weights = True
        else:
            logger.warning(
                f"No checkpoints found in {work_dir}. Inference may use stale weights."
            )

        # Update persistent configuration context
        self._source_toml = (work_dir / "user_config.toml").absolute()
        self._source_dir = work_dir.absolute()
        self.config_path = config_path

        # Refresh metadata (critical for first-time training of custom models)
        meta = self._config_manager.load_metadata_from_toml(self._source_toml)
        self.num_classes = meta.get("num_classes")
        self.num_keypoints = meta.get("num_keypoints")
        self.metainfo = meta.get("metainfo")
        self.architecture_params = meta.get("architecture_params", {})

        # Clear inferencer to force re-initialization on next predict() call
        if hasattr(self, "_inferencer"):
            self._inferencer = None

    def _run_training(self, config_path: Path, log_level: str) -> None:
        """Initializes the MMEngine Runner and starts training."""
        logger.info(f"Starting MMEngine Runner with config: {config_path}")
        runner = Runner.from_cfg(Config.fromfile(str(config_path)))
        runner.train()

    def _load_base_config(self, model: str) -> Config:
        config_path = get_config_file(model)

        with switch_to_lib_root(self.model) as lib_root:
            # Use relative path from lib_root
            rel_config_path = config_path.relative_to(lib_root)
            cfg = Config.fromfile(str(rel_config_path))

        return cfg

    def _inject_user_configs(self, config: UserConfig) -> None:
        """Applies configuration changes using the registered plugin surgeries."""
        if not self._cfg:
            raise RuntimeError("Base config not loaded.")

        for surgery in get_surgeries(self.model):
            surgery.apply(self._cfg, config)
