import tempfile
import importlib.util
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from loguru import logger
from mmengine.config import Config

from ez_openmmlab.core.resolvers import ModelParamsResolverFactory
from ez_openmmlab.core.schema.datasets import DatasetConfig
from ez_openmmlab.core.schema.models import ModelName
from ez_openmmlab.core.schema import config as toml_config
from ez_openmmlab.core.surgery import get_surgeries
from ez_openmmlab.core.utils.context import switch_to_lib_root


class BaseConfigLoader:
    """Resolves model names to absolute paths of official OpenMMLab config files."""

    def __init__(self):
        # Dynamically resolve mmdet and mmpose installed locations
        self._mmdet_config_root = self._resolve_package_configs("mmdet")
        self._mmpose_config_root = self._resolve_package_configs("mmpose")

        logger.debug(
            f"Config Roots: mmdet={self._mmdet_config_root}, mmpose={self._mmpose_config_root}"
        )
        self._validate_root()

    def _resolve_package_configs(self, package_name: str) -> Optional[Path]:
        """Finds the .mim/configs path for an installed OpenMMLab package."""
        spec = importlib.util.find_spec(package_name)
        if not spec or not spec.origin:
            return None

        package_root = Path(spec.origin).parent
        config_path = package_root / ".mim" / "configs"

        if config_path.exists():
            return config_path

        # Fallback for standard 'configs' directory if .mim is not used
        legacy_config_path = package_root / "configs"
        if legacy_config_path.exists():
            return legacy_config_path

        return None

    def _validate_root(self):
        """Ensures at least one config root exists."""
        if (not self._mmdet_config_root or not self._mmdet_config_root.exists()) and (
            not self._mmpose_config_root or not self._mmpose_config_root.exists()
        ):
            logger.error("Could not find installed OpenMMLab configs.")
            raise FileNotFoundError(
                "Could not find installed mmdetection or mmpose configs.\n"
                "Ensure they are installed in your environment."
            )

    def get_config_path(self, model_name: str | ModelName) -> Path:
        """Resolves a model name to its absolute config path."""
        actual_name = (
            model_name.value if isinstance(model_name, ModelName) else model_name
        )
        try:
            model = ModelName(actual_name)
            rel_path = model.config_path
        except ValueError:
            logger.error(f"Model '{actual_name}' is not supported or recognized.")
            supported = ", ".join([m.value for m in ModelName])
            raise ValueError(
                f"Model '{actual_name}' not found in internal map.\n"
                f"Currently supported models: {supported}"
            )

        # Determine which library root to use
        if "rtmpose" in actual_name or "rtmo" in actual_name:
            config_root = self._mmpose_config_root
        else:
            config_root = self._mmdet_config_root

        config_path = config_root / rel_path

        if not config_path.exists():
            logger.error(f"Config file for '{actual_name}' missing at: {config_path}")
            raise FileNotFoundError(
                f"Config file for '{actual_name}' not found at: {config_path}\n"
                "Please verify the appropriate OpenMMLab submodule is correctly initialized."
            )

        logger.debug(f"Resolved model '{actual_name}' to: {config_path}")
        return config_path


# Global internal instance for BaseConfigLoader logic
_LOADER = BaseConfigLoader()


def get_config_file(model_name: str | ModelName) -> Path:
    """Public utility to get the absolute path to a base model's config file."""
    return _LOADER.get_config_path(
        str(model_name.value) if isinstance(model_name, ModelName) else model_name
    )


class ConfigManager:
    """Consolidated manager for constructing UserConfig and handling temporary config lifecycle."""

    def load_base_config(self, model_name: str, config_path: Path) -> Config:
        """Loads the base OpenMMLab configuration file.

        Args:
            model_name: Name of the model (for context switching).
            config_path: Absolute path to the .py config file.
        """
        # If the path is absolute and exists, we can try loading it directly
        # especially for temporary/mocked configs in tests.
        if config_path.is_absolute() and config_path.exists():
            try:
                # We still want the context switch for any relative imports inside the config
                with switch_to_lib_root(model_name):
                    return Config.fromfile(str(config_path))
            except Exception:
                # Fallback to relative resolution if direct load fails
                pass

        with switch_to_lib_root(model_name) as lib_root:
            try:
                rel_config_path = config_path.relative_to(lib_root)
                return Config.fromfile(str(rel_config_path))
            except ValueError:
                import sys

                logger.error(
                    f"Configuration file '{config_path.name}' was not found in the "
                    f"designated OpenMMLab location: {lib_root}\n"
                )
                sys.exit(1)

    def patch_config(
        self, cfg: Config, model_name: str, user_config: toml_config.UserConfig
    ) -> None:
        """Applies all registered plugin surgeries to a Config object."""
        for surgery in get_surgeries(model_name):
            surgery.apply(cfg, user_config)

    def get_dummy_user_config(
        self,
        model_name: str,
        num_classes: Optional[int] = None,
        num_keypoints: Optional[int] = None,
        architecture_params: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> toml_config.UserConfig:
        """Creates a dummy UserConfig to satisfy the injector interface.

        This is used for on-the-fly patching during inference or export.
        """
        model_params = {
            "name": model_name,
            "num_classes": num_classes,
            "num_keypoints": num_keypoints,
        }
        if architecture_params:
            model_params.update(architecture_params)
        model_params.update(kwargs)

        return toml_config.UserConfig(
            model=toml_config.ModelSection(**model_params),
            training=toml_config.TrainingSection(num_workers=0, learning_rate=0.001),
            data=toml_config.DataSection(root=""),
        )

    def create_fresh_config(
        self,
        model: str,
        dataset_config_path: Union[str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        epochs: int = 100,
        batch_size: int = 8,
        device: str = "cuda",
        work_dir: str = "./runs/train",
        learning_rate: float = 0.001,
        amp: bool = True,
        num_workers: int = 4,
        enable_tensorboard: bool = False,
        log_level: str = "INFO",
        # Core training hyperparameters
        weight_decay: float = 0.05,
        evaluator_metric: Union[str, List[str]] = "CocoMetric",
        resume: Union[bool, str] = False,
        stage2_num_epochs: int = 20,
        architecture_params: Optional[Dict[str, Any]] = None,
        # Augmentation parameters
        scale_factor: Optional[Union[float, Tuple[float, float], List[float]]] = None,
        rotate_factor: Optional[float] = None,
        random_flip_prob: Optional[float] = None,
        **kwargs,
    ) -> toml_config.UserConfig:
        """Assembles a brand new UserConfig object from training parameters and dataset TOML."""
        dataset_cfg = DatasetConfig.from_toml(Path(dataset_config_path))

        dataset_name = dataset_cfg.dataset_name
        classes = dataset_cfg.classes
        if not classes:
            raise ValueError(
                f"No 'classes' defined in dataset configuration: {dataset_config_path}. "
                "Explicit class names are required for all datasets."
            )
        num_classes = len(classes)

        # Extract num_keypoints from metainfo if available (for pose models)
        num_keypoints = None
        if dataset_cfg.metainfo and "keypoint_info" in dataset_cfg.metainfo:
            num_keypoints = len(dataset_cfg.metainfo["keypoint_info"])

        # --- Delegate Parameter Resolution ---
        model_params_resolver = ModelParamsResolverFactory.get_resolver(model)
        if model_params_resolver:
            resolved_model_params = model_params_resolver.resolve(
                **(architecture_params or {})
            )
        else:
            resolved_model_params = architecture_params or {}

        # Build ModelSection parameters dynamically
        model_params = {
            "name": model,
            "num_classes": num_classes,
            "num_keypoints": num_keypoints,
            "load_from": str(checkpoint_path) if checkpoint_path else None,
        }
        model_params.update(resolved_model_params)

        return toml_config.UserConfig(
            model=toml_config.ModelSection(**model_params),
            data=toml_config.DataSection(
                root=str(dataset_cfg.data_root),
                dataset_name=dataset_name,
                train_ann=dataset_cfg.train.ann_file,
                train_img=dataset_cfg.train.img_dir,
                val_ann=dataset_cfg.val.ann_file,
                val_img=dataset_cfg.val.img_dir,
                test_ann=dataset_cfg.test.ann_file if dataset_cfg.test else None,
                test_img=dataset_cfg.test.img_dir if dataset_cfg.test else None,
                classes=classes,
                metainfo=dataset_cfg.metainfo,
            ),
            training=toml_config.TrainingSection(
                epochs=epochs,
                batch_size=batch_size,
                learning_rate=learning_rate,
                weight_decay=weight_decay,
                device=device,
                work_dir=work_dir,
                log_level=log_level,
                amp=amp,
                num_workers=num_workers,
                enable_tensorboard=enable_tensorboard,
                evaluator_metric=evaluator_metric,
                resume=resume,
                stage2_num_epochs=stage2_num_epochs,
            ),
            augments=toml_config.AugmentationSection(
                scale_factor=scale_factor,
                rotate_factor=rotate_factor,
                random_flip_prob=random_flip_prob,
            ),
        )

    def recover_config_from_toml(
        self,
        toml_path: Path,
        **kwargs,
    ) -> toml_config.UserConfig:
        """Loads an existing UserConfig and applies training parameter overrides."""
        user_cfg = toml_config.load_user_config(toml_path)

        # Update training parameters with any explicit overrides provided to train()
        training_fields = [
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "device",
            "work_dir",
            "log_level",
            "amp",
            "num_workers",
            "enable_tensorboard",
            "evaluator_metric",
            "resume",
            "stage2_num_epochs",
        ]

        for field in training_fields:
            if field in kwargs and kwargs[field] is not None:
                setattr(user_cfg.training, field, kwargs[field])

        # Also allow overriding augmentations during resume if passed
        if "augments" in kwargs and isinstance(kwargs["augments"], dict):
            for k, v in kwargs["augments"].items():
                if hasattr(user_cfg.augments, k):
                    setattr(user_cfg.augments, k, v)

        logger.debug(f"Recovered and updated resume configuration from: {toml_path}")
        return user_cfg

    def load_dataset_for_export(
        self, source_toml: Path, path_prefix: Optional[str] = None
    ) -> Optional[toml_config.DataSection]:
        """Loads dataset config from TOML and optionally adjusts paths for Docker."""
        if not source_toml.exists():
            return None

        try:
            # Reusing recover_config_from_toml to parse the TOML
            user_cfg = self.recover_config_from_toml(source_toml)
            data = user_cfg.data

            if path_prefix:
                # Adjust root path if it's relative
                if not Path(data.root).is_absolute():
                    data.root = f"{path_prefix}/{data.root}"
                    logger.debug(f"Adjusted data root for export: {data.root}")

            return data
        except Exception as e:
            logger.warning(f"Failed to load dataset for export from {source_toml}: {e}")
            return None

    def load_metadata_from_toml(self, config_path: Path) -> Dict[str, Any]:
        """Extracts training metadata from a user_config.toml file."""
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        metadata = {
            "num_classes": None,
            "num_keypoints": None,
            "metainfo": None,
            "model_name": None,
            "architecture_params": {},
            "augmentation_params": {},
        }

        try:
            user_cfg = toml_config.load_user_config(config_path)
            metadata["num_classes"] = user_cfg.model.num_classes
            metadata["num_keypoints"] = user_cfg.model.num_keypoints
            metadata["metainfo"] = user_cfg.data.metainfo

            if user_cfg.model.model_extra:
                metadata["architecture_params"] = user_cfg.model.model_extra

            metadata["augmentation_params"] = user_cfg.augments.model_dump(
                exclude_none=True
            )

            if user_cfg.data.classes:
                if metadata["metainfo"] is None:
                    metadata["metainfo"] = {}
                metadata["metainfo"]["classes"] = user_cfg.data.classes

            metadata["model_name"] = user_cfg.model.name.value
            logger.debug(f"Loaded metadata from: {config_path}")
        except Exception as e:
            logger.warning(f"Failed to load metadata from {config_path}: {e}")

        return metadata

    def prepare_config_file(
        self, toml_path: Path, checkpoint_path: Optional[Path] = None
    ) -> Path:
        """Generates a temporary python config file from a custom config.toml."""
        if not toml_path.exists():
            raise FileNotFoundError(f"Custom config file not found: {toml_path}")

        user_cfg = toml_config.load_user_config(toml_path)

        # Resolve base config
        base_config_path = get_config_file(user_cfg.model.name.value)
        base_config_str = str(base_config_path.absolute())

        # Create content for temp config
        content = f'_base_ = ["{base_config_str}"]\n'

        # Create temp file
        temp_file = tempfile.NamedTemporaryFile(suffix=".py", mode="w", delete=False)
        temp_file.write(content)
        temp_file.close()

        logger.debug(f"Created temporary config file: {temp_file.name}")
        return Path(temp_file.name)

    def cleanup_temp_config(self, config_path: Optional[Path]) -> None:
        """Removes a temporary config file if it exists."""
        if config_path and config_path.exists():
            try:
                config_path.unlink()
                logger.debug(f"Removed temporary config file: {config_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temp config file {config_path}: {e}")

    def dump_config(self, cfg: Config, output_path: Path) -> Path:
        """Saves a memory Config object to a self-contained .py file."""
        cfg.dump(str(output_path))
        logger.debug(f"Flattened configuration saved to: {output_path}")
        return output_path
