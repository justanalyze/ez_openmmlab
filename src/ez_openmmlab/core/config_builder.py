from pathlib import Path
from typing import Optional, Union, Dict, Any, List
import tempfile
from loguru import logger

from ez_openmmlab.core.config_loader import get_config_file
from ez_openmmlab.schemas.dataset import DatasetConfig
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils.toml_config import (
    DataSection,
    ModelSection,
    TrainingSection,
    UserConfig,
    load_user_config,
)


class UserConfigBuilder:
    """Utility class for constructing and managing UserConfig objects and temporary files."""

    def build(
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
    ) -> UserConfig:
        """Assembles a full UserConfig object from training parameters and dataset TOML."""
        dataset_cfg = DatasetConfig.from_toml(Path(dataset_config_path))

        classes = dataset_cfg.classes
        num_classes = len(classes) if classes else 80

        # Extract num_keypoints from metainfo if available (for pose models)
        num_keypoints = None
        if dataset_cfg.metainfo and "keypoint_info" in dataset_cfg.metainfo:
            num_keypoints = len(dataset_cfg.metainfo["keypoint_info"])

        return UserConfig(
            model=ModelSection(
                name=model,
                num_classes=num_classes,
                num_keypoints=num_keypoints,
                load_from=str(checkpoint_path) if checkpoint_path else None,
            ),
            data=DataSection(
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
            training=TrainingSection(
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

    def load_metadata_from_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """Attempts to find and load training metadata from nearby config files."""
        search_dirs = [checkpoint_path.parent, checkpoint_path.parent.parent]
        config_files = ["user_config.toml", "dataset.toml"]

        metadata = {
            "num_classes": None,
            "num_keypoints": None,
            "metainfo": None,
            "model_name": None,
        }

        for d in search_dirs:
            for f in config_files:
                path = d / f
                if path.exists():
                    try:
                        if f == "user_config.toml":
                            user_cfg = load_user_config(path)
                            metadata["num_classes"] = user_cfg.model.num_classes
                            metadata["num_keypoints"] = user_cfg.model.num_keypoints
                            metadata["metainfo"] = user_cfg.data.metainfo
                            metadata["model_name"] = user_cfg.model.name.value
                            logger.info(f"Auto-loaded metadata from: {path}")
                        else:
                            ds_cfg = DatasetConfig.from_toml(path)
                            metadata["num_classes"] = (
                                len(ds_cfg.classes) if ds_cfg.classes else None
                            )
                            metadata["metainfo"] = ds_cfg.metainfo
                            if ds_cfg.metainfo and "keypoint_info" in ds_cfg.metainfo:
                                metadata["num_keypoints"] = len(
                                    ds_cfg.metainfo["keypoint_info"]
                                )
                            logger.info(f"Auto-loaded metadata from: {path}")
                        return metadata
                    except Exception as e:
                        logger.warning(f"Failed to auto-load metadata from {path}: {e}")
        return metadata

    def prepare_config_file(
        self, toml_path: Path, checkpoint_path: Optional[Path] = None
    ) -> Path:
        """Generates a temporary python config file from a custom config.toml."""
        if not toml_path.exists():
            raise FileNotFoundError(f"Custom config file not found: {toml_path}")

        user_cfg = load_user_config(toml_path)

        # Resolve base config
        base_config_path = get_config_file(user_cfg.model.name.value)
        base_config_str = str(base_config_path.absolute())

        # Create content for temp config
        content = f'_base_ = ["{base_config_str}"]\n'

        # Create temp file
        # We use NamedTemporaryFile but don't delete on close so it can be used by OpenMMLab
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
