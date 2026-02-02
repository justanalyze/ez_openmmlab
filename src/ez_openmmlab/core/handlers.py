from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any

from mmengine.config import Config

from ez_openmmlab.utils.toml_config import UserConfig


class BaseConfigHandler(ABC):
    """Abstract base class for configuration handlers.
    Each handler is responsible for configuring a specific aspect of the MMDetection config.
    """

    @abstractmethod
    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        """Applies configuration updates to the MMDetection Config object.

        Args:
            cfg: The mutable MMDetection Config object.
            user_config: The validated user configuration.
        """
        pass


class DataloaderHandler(BaseConfigHandler):
    """Configures dataset paths, batch sizes, and workers for train/val/test loaders."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        data_root = Path(user_config.data.root)
        cfg.data_root = str(data_root)

        # 1. Configure each dataloader (train/val/test)
        for key in ["train_dataloader", "val_dataloader", "test_dataloader"]:
            if hasattr(cfg, key):
                self._configure_single_loader(getattr(cfg, key), key, data_root, user_config)

        # 2. Configure global metainfo at the config root
        self._configure_global_metainfo(cfg, user_config)

    def _configure_single_loader(
        self, dl: Config, loader_name: str, data_root: Path, user_config: UserConfig
    ) -> None:
        """Configures a specific dataloader instance."""
        # Prevent double-joining by clearing internal data_root
        dl.dataset.data_root = ""
        dl.batch_size = user_config.training.batch_size
        dl.num_workers = user_config.training.num_workers
        dl.persistent_workers = user_config.training.num_workers > 0

        # Resolve absolute paths for annotations and images
        if loader_name == "train_dataloader":
            ann_path = user_config.data.train_ann
            img_path = user_config.data.train_img
        else:
            # Fallback for test/val sharing validation set
            ann_path = user_config.data.val_ann
            img_path = user_config.data.val_img

        dl.dataset.ann_file = str(data_root / ann_path)
        dl.dataset.data_prefix = {"img": str(data_root / img_path)}

        # Update metainfo for this dataset instance
        self._apply_metainfo(dl.dataset, user_config)

    def _configure_global_metainfo(self, cfg: Config, user_config: UserConfig) -> None:
        """Applies metainfo overrides to the root config object."""
        self._apply_metainfo(cfg, user_config)

    def _apply_metainfo(self, target_cfg: Config, user_config: UserConfig) -> None:
        """Injects classes and metainfo into a target config section."""
        if user_config.data.classes:
            target_cfg.metainfo = {"classes": user_config.data.classes}

        if user_config.data.metainfo:
            if not hasattr(target_cfg, "metainfo") or target_cfg.metainfo is None:
                target_cfg.metainfo = {}
            
            # Process and merge detailed metainfo (keypoints, skeletons, etc.)
            processed = self._process_metainfo_keys(user_config.data.metainfo)
            target_cfg.metainfo.update(processed)

    @staticmethod
    def _process_metainfo_keys(metainfo: Dict[str, Any]) -> Dict[str, Any]:
        """Converts string keys to integers for specific fields (TOML compatibility)."""
        processed = metainfo.copy()
        for key in ["keypoint_info", "skeleton_info"]:
            if key in processed and isinstance(processed[key], dict):
                processed[key] = {
                    int(k) if k.isdigit() else k: v 
                    for k, v in processed[key].items()
                }
        return processed


class RuntimeHandler(BaseConfigHandler):
    """Configures general runtime settings including optimizer, AMP, and visualization."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        self._configure_general(cfg, user_config)
        self._configure_optimizer(cfg, user_config)
        self._configure_visualization(cfg, user_config)
        self._configure_evaluators(cfg, user_config)

    def _configure_general(self, cfg: Config, user_config: UserConfig) -> None:
        cfg.work_dir = user_config.training.work_dir
        cfg.train_cfg.max_epochs = user_config.training.epochs
        cfg.load_from = user_config.model.load_from
        cfg.log_level = user_config.training.log_level

    def _configure_optimizer(self, cfg: Config, user_config: UserConfig) -> None:
        if not hasattr(cfg, "optim_wrapper"):
            return

        training = user_config.training
        
        # Configure AMP (Automatic Mixed Precision)
        if training.amp:
            cfg.optim_wrapper.type = "AmpOptimWrapper"
            cfg.optim_wrapper.loss_scale = "dynamic"
        else:
            cfg.optim_wrapper.type = "OptimWrapper"
            if hasattr(cfg.optim_wrapper, "loss_scale"):
                del cfg.optim_wrapper["loss_scale"]

        # Set Learning Rate
        if hasattr(cfg.optim_wrapper, "optimizer"):
            cfg.optim_wrapper.optimizer.lr = training.learning_rate

    def _configure_visualization(self, cfg: Config, user_config: UserConfig) -> None:
        if not user_config.training.enable_tensorboard:
            return

        # Ensure visualizer structure exists
        if not hasattr(cfg, "visualizer"):
            cfg.visualizer = dict(
                type="DetLocalVisualizer",
                vis_backends=[dict(type="LocalVisBackend")],
            )

        if "vis_backends" not in cfg.visualizer:
            cfg.visualizer["vis_backends"] = [dict(type="LocalVisBackend")]

        # Idempotently add Tensorboard backend
        vis_backends = cfg.visualizer["vis_backends"]
        if not any(b["type"] == "TensorboardVisBackend" for b in vis_backends):
            vis_backends.append(dict(type="TensorboardVisBackend"))

    def _configure_evaluators(self, cfg: Config, user_config: UserConfig) -> None:
        data_root = Path(user_config.data.root)
        ann_path = str(data_root / user_config.data.val_ann)

        if hasattr(cfg, "val_evaluator"):
            cfg.val_evaluator.ann_file = ann_path
        
        if hasattr(cfg, "test_evaluator"):
            # Fallback to validation set for now if explicit test set isn't defined
            cfg.test_evaluator.ann_file = ann_path