import cv2
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from loguru import logger
from mmengine.config import Config

from ez_openmmlab.core.config_manager import ConfigManager
from ez_openmmlab.core.inference.results import InferenceResult
from ez_openmmlab.core.resolvers import ResourceResolver
from ez_openmmlab.core.validators import InputValidator
from ez_openmmlab.core.training.orchestrator import TrainingOrchestrator
from ez_openmmlab.core.schema.models import ModelName
from ez_openmmlab.core.utils.input import normalize_inputs
from ez_openmmlab.core.utils.path import get_unique_dir
from ez_openmmlab.core.deploy.config_modifier import DeployConfigModifier


class EZMMLab(ABC):
    """Abstract base class for all OpenMMLab models.

    Acts as a high-level coordinator, delegating specific tasks to specialized
    components like ResourceResolver, ConfigManager, and TrainingOrchestrator.
    """

    def __init__(
        self,
        model: Union[ModelName, str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
        **kwargs,
    ):
        # --- 1. Logging ---
        self.log_level = log_level
        self._configure_logging(log_level)
        logger.info(f"Initializing {self.__class__.__name__} with model: '{model}'")

        # Ensure noisy warnings are suppressed immediately
        from ez_openmmlab import mute_warnings

        mute_warnings()

        # --- 2. Internal Managers ---
        self._config_manager = ConfigManager()
        self._resource_resolver = ResourceResolver(self._config_manager)
        self._training_orchestrator = TrainingOrchestrator()

        # --- 3. Resource Resolution ---
        resources = self._resource_resolver.resolve(model, checkpoint_path)

        self.checkpoint_path = resources.checkpoint_path
        self.config_path = resources.config_path
        self.model = resources.model_name
        self.num_classes = kwargs.get("num_classes") or resources.num_classes
        self.num_keypoints = kwargs.get("num_keypoints") or resources.num_keypoints
        self.metainfo = kwargs.get("metainfo") or resources.metainfo
        self.architecture_params = resources.architecture_params

        self._cfg: Optional[Config] = None
        self._temp_config_file = resources.temp_config_file
        self._source_dir = resources.source_dir
        self._source_toml = resources.source_toml
        self._using_custom_weights = checkpoint_path is not None

        # --- 4. Validation ---
        InputValidator.validate_initialization(
            model=model,
            checkpoint_path=self.checkpoint_path,
            using_custom_weights=self._using_custom_weights,
            num_classes=self.num_classes,
            num_keypoints=self.num_keypoints,
        )

    def _configure_logging(self, log_level: str) -> None:
        """Configures the global logger level and suppresses noisy dependencies."""
        try:
            import sys

            logger.remove()
            logger.add(
                sys.stderr,
                level=log_level,
                filter=lambda record: record["level"].no >= logger.level(log_level).no,
            )
            import logging

            logging.getLogger("mmengine").setLevel(
                logging.ERROR if log_level == "INFO" else log_level
            )
        except Exception as e:
            print(f"Warning: Failed to set log level: {e}")

    def __del__(self):
        """Cleanup temporary files."""
        if hasattr(self, "_temp_config_file") and self._temp_config_file:
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
        """Universal prediction entry point."""
        if not self.checkpoint_path:
            raise ValueError("Inference requires a valid weight file (.pth).")

        merged_kwargs = {**self.architecture_params, **kwargs}
        architecture_params = self._get_architecture_params(**merged_kwargs)

        self._init_inferencer(device, **{**architecture_params, **kwargs})

        actual_out_dir = str(get_unique_dir(out_dir)) if out_dir else ""
        inputs = normalize_inputs(image_path)

        if not hasattr(self, "_inferencer") or self._inferencer is None:
            raise RuntimeError("Inferencer failed to initialize.")

        raw_results = self._run_inference(inputs, actual_out_dir, show, **kwargs)

        if show:
            cv2.destroyAllWindows()

        if not hasattr(self, "_formatter") or self._formatter is None:
            raise RuntimeError("Result formatter not initialized.")

        return self._formatter.map_results(raw_results, inputs, self._get_class_names())

    def export(
        self,
        format: str,
        image: Union[str, Path],
        output_dir: str = "runs/deploy",
        device: str = "cpu",
        **kwargs,
    ) -> Path:
        """Exports the model to production formats via Docker."""
        if not self.checkpoint_path or not self.checkpoint_path.exists():
            raise ValueError("Export requires a valid checkpoint path.")

        if format == "tensorrt" and device == "cpu":
            raise ValueError("TensorRT export requires a GPU (device='cuda').")

        from ez_openmmlab.core.deploy.docker_manager import DockerExportManager
        from ez_openmmlab.core.deploy.registry import DeployConfigRegistry

        current_path = Path(__file__).resolve()
        project_root = next(
            (p for p in current_path.parents if (p / "pyproject.toml").exists()), None
        )

        if not project_root:
            raise RuntimeError("Could not determine project root.")

        registry = DeployConfigRegistry()
        deploy_cfg = registry.get_deploy_cfg(
            self._get_library_family(), format, model_name=self.model
        )

        manager = DockerExportManager(project_root=project_root)
        image_tag = kwargs.pop("image_tag", "ubuntu20.04-cuda11.8-mmdeploy1.3.1")

        # Ensure Docker is available and the image exists (prompting if needed)
        manager.ensure_image_available(image_tag)

        export_work_dir = Path(output_dir)
        export_work_dir.mkdir(parents=True, exist_ok=True)

        # Generate custom deploy config if input_size
        # is specified (e.g. from user_config.toml)
        input_size = self.architecture_params.get("input_size")
        if (
            input_size
            and isinstance(input_size, (list, tuple))
            and len(input_size) == 2
        ):
            deploy_cfg = DeployConfigModifier.generate_input_resize_config(
                base_deploy_cfg=deploy_cfg,
                input_size=input_size,
                output_dir=export_work_dir,
            )

        # Use expanded ConfigManager for loading and patching
        patched_cfg = self._load_and_patch_config(docker_mode=True, **kwargs)
        final_cfg_path = export_work_dir / "config.py"
        self._config_manager.dump_config(patched_cfg, final_cfg_path)

        cmd = manager.build_command(
            deploy_cfg=deploy_cfg,
            model_cfg=str(final_cfg_path.absolute()),
            checkpoint=str(self.checkpoint_path.absolute()),
            test_img=str(Path(image).absolute()),
            work_dir=output_dir,
            device=device,
            image_tag=image_tag,
            **kwargs,
        )

        manager.run_export(cmd)
        return Path(output_dir)

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
        **kwargs,
    ) -> None:
        """Runs a fresh end-to-end training pipeline."""
        logger.info(f"Assembling fresh training config for: {dataset_config_path}")
        InputValidator.validate_augments(
            augments, self._get_library_family(), self.__class__.__name__
        )

        architecture_params = self._get_architecture_params(**kwargs)
        aug_dict = augments or {}

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
            scale_factor=aug_dict.get("scale_factor"),
            rotate_factor=aug_dict.get("rotate_factor"),
            random_flip_prob=aug_dict.get("random_flip_prob"),
            stage2_num_epochs=stage2_num_epochs,
            **kwargs,
        )

        self._training_orchestrator.run(self, user_config, dry_run=dry_run)

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
        """Resumes an unfinished training session."""
        if not self._source_toml:
            raise ValueError(
                "Resumption requires initialization with 'user_config.toml'."
            )

        effective_work_dir = work_dir or str(self._source_dir)
        user_config = self._config_manager.recover_config_from_toml(
            toml_path=self._source_toml,
            resume=checkpoint,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            work_dir=effective_work_dir,
            stage2_num_epochs=stage2_num_epochs,
            **kwargs,
        )

        self._training_orchestrator.run(self, user_config, dry_run=dry_run)

    def _load_and_patch_config(self, **kwargs) -> Config:
        """Loads and patches the OpenMMLab config (delegated to ConfigManager)."""
        cfg = self._config_manager.load_base_config(self.model, self.config_path)
        dummy_user_cfg = self._config_manager.get_dummy_user_config(
            self.model,
            self.num_classes,
            self.num_keypoints,
            self.architecture_params,
            **kwargs,
        )

        # Attempt to recover dataset info from source TOML if available
        if (
            hasattr(self, "_source_toml")
            and self._source_toml
            and self._source_toml.exists()
        ):
            path_prefix = "/work" if kwargs.get("docker_mode", False) else None
            data_section = self._config_manager.load_dataset_for_export(
                self._source_toml, path_prefix=path_prefix
            )
            if data_section:
                # For inference, we don't want to inject the training dataset type
                # as it might not be registered or needed.
                if not kwargs.get("docker_mode", False):
                    data_section.dataset_name = None
                dummy_user_cfg.data = data_section

        self._config_manager.patch_config(cfg, self.model, dummy_user_cfg)
        return cfg

    def _get_class_names(self) -> dict:
        """Retrieves class names from local metainfo or inferencer."""
        if self.metainfo and "classes" in self.metainfo:
            return {i: name for i, name in enumerate(self.metainfo["classes"])}

        if hasattr(self, "_inferencer") and self._inferencer:
            model = getattr(self._inferencer, "model", None)
            if model:
                meta = getattr(model, "dataset_meta", {})
                if "classes" in meta:
                    return {i: name for i, name in enumerate(meta["classes"])}
        return {}

    @abstractmethod
    def _init_inferencer(self, device: str, **kwargs) -> None:
        pass

    @abstractmethod
    def _run_inference(
        self, inputs: list, out_dir: str, show: bool, **kwargs
    ) -> Union[dict, list]:
        pass

    @abstractmethod
    def _get_library_family(self) -> str:
        pass

    @abstractmethod
    def _get_architecture_params(self, **kwargs) -> Dict[str, Any]:
        pass
