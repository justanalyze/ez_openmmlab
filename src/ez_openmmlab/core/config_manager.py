import tempfile
from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger

from ez_openmmlab.schemas.dataset import DatasetConfig
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils import toml_config


class BaseConfigLoader:
    """Resolves model names to absolute paths of official OpenMMLab config files."""

    def __init__(self):
        # Resolve relative to the installed package location
        # Structure: ez_openmmlab/core/config_manager.py -> ez_openmmlab/../../libs
        self._project_root = Path(__file__).resolve().parents[3]
        self._mmdet_config_root = (
            self._project_root / "libs" / "mmdetection" / "configs"
        )
        self._mmpose_config_root = self._project_root / "libs" / "mmpose" / "configs"

        logger.debug(
            f"Initializing BaseConfigLoader. Project root: {self._project_root}"
        )
        self._validate_root()

    def _validate_root(self):
        """Ensures at least one config root exists."""
        if (
            not self._mmdet_config_root.exists()
            and not self._mmpose_config_root.exists()
        ):
            logger.error("Could not find local OpenMMLab configs.")
            raise FileNotFoundError(
                "Could not find local mmdetection or mmpose configs.\n"
                "Ensure submodules are initialized."
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

    def build_user_config(
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
    ) -> toml_config.UserConfig:
        """Assembles a full UserConfig object from training parameters and dataset TOML."""
        dataset_cfg = DatasetConfig.from_toml(Path(dataset_config_path))

        classes = dataset_cfg.classes
        num_classes = len(classes) if classes else 80

        # Extract num_keypoints from metainfo if available (for pose models)
        num_keypoints = None
        if dataset_cfg.metainfo and "keypoint_info" in dataset_cfg.metainfo:
            num_keypoints = len(dataset_cfg.metainfo["keypoint_info"])

        return toml_config.UserConfig(
            model=toml_config.ModelSection(
                name=model,
                num_classes=num_classes,
                num_keypoints=num_keypoints,
                load_from=str(checkpoint_path) if checkpoint_path else None,
            ),
            data=toml_config.DataSection(
                root=str(dataset_cfg.data_root),
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
                device=device,
                work_dir=work_dir,
                log_level=log_level,
                amp=amp,
                num_workers=num_workers,
                enable_tensorboard=enable_tensorboard,
            ),
        )

    def load_metadata_from_toml(self, config_path: Path) -> Dict[str, Any]:
        """Extracts training metadata from a user_config.toml file.

        Args:
            config_path: Path to the user_config.toml file.

        Returns:
            A dictionary containing model_name, num_classes, num_keypoints, and metainfo.
        """
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        metadata = {
            "num_classes": None,
            "num_keypoints": None,
            "metainfo": None,
            "model_name": None,
        }

        try:
            user_cfg = toml_config.load_user_config(config_path)
            metadata["num_classes"] = user_cfg.model.num_classes
            metadata["num_keypoints"] = user_cfg.model.num_keypoints
            metadata["metainfo"] = user_cfg.data.metainfo

            # Explicitly include classes in metainfo if defined in DataSection
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
