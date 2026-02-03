from typing import Any, Dict
from mmengine.config import Config
from ez_openmmlab.utils.toml_config import UserConfig
from .base import BaseConfigHandler

class DataloaderHandler(BaseConfigHandler):
    """Configures dataset paths, batch sizes, and workers for all loaders."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        cfg.data_root = user_config.data.root
        metainfo = self._assemble_metainfo(user_config)

        if metainfo:
            cfg.metainfo = metainfo

        for name in ["train_dataloader", "val_dataloader", "test_dataloader"]:
            if hasattr(cfg, name):
                self._configure_dataloader(getattr(cfg, name), name, user_config, metainfo)

    def _assemble_metainfo(self, user_config: UserConfig) -> Dict[str, Any]:
        metainfo = {}
        if user_config.data.classes:
            metainfo["classes"] = user_config.data.classes
        if user_config.data.metainfo:
            metainfo.update(self._process_metainfo(user_config.data.metainfo))
        return metainfo

    def _configure_dataloader(self, dl: Config, name: str, user_config: UserConfig, metainfo: Dict[str, Any]) -> None:
        dl.dataset.data_root = ""
        dl.batch_size = user_config.training.batch_size
        dl.num_workers = user_config.training.num_workers
        dl.persistent_workers = user_config.training.num_workers > 0

        if name == "train_dataloader":
            dl.dataset.ann_file = user_config.data.train_ann_path
            dl.dataset.data_prefix = {"img": user_config.data.train_img_path}
        else:
            dl.dataset.ann_file = user_config.data.val_ann_path
            dl.dataset.data_prefix = {"img": user_config.data.val_img_path}

        if metainfo:
            dl.dataset.metainfo = metainfo

class RuntimeHandler(BaseConfigHandler):
    """Configures optimizer, AMP, epochs, and basic logging."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        cfg.work_dir = user_config.training.work_dir
        cfg.train_cfg.max_epochs = user_config.training.epochs
        cfg.load_from = user_config.model.load_from
        cfg.log_level = user_config.training.log_level

        if hasattr(cfg, "optim_wrapper"):
            training = user_config.training
            if training.amp:
                cfg.optim_wrapper.type = "AmpOptimWrapper"
                cfg.optim_wrapper.loss_scale = "dynamic"
            else:
                cfg.optim_wrapper.type = "OptimWrapper"
                cfg.optim_wrapper.pop("loss_scale", None)

            if hasattr(cfg.optim_wrapper, "optimizer"):
                cfg.optim_wrapper.optimizer.lr = training.learning_rate

        self._configure_visualization(cfg, user_config)
        self._configure_evaluator_paths(cfg, user_config)

    def _configure_visualization(self, cfg: Config, user_config: UserConfig) -> None:
        """Sets up visualization backends (Local, TensorBoard)."""
        if not user_config.training.enable_tensorboard:
            return

        # Ensure visualizer base structure exists
        if not hasattr(cfg, "visualizer"):
            cfg.visualizer = dict(
                type="DetLocalVisualizer",
                vis_backends=[dict(type="LocalVisBackend")],
            )

        if "vis_backends" not in cfg.visualizer:
            cfg.visualizer["vis_backends"] = [dict(type="LocalVisBackend")]

        # Add Tensorboard if not already present
        vis_backends = cfg.visualizer["vis_backends"]
        if not any(b["type"] == "TensorboardVisBackend" for b in vis_backends):
            vis_backends.append(dict(type="TensorboardVisBackend"))

    def _configure_evaluator_paths(self, cfg: Config, user_config: UserConfig) -> None:
        if hasattr(cfg, "val_evaluator"):
            cfg.val_evaluator.ann_file = user_config.data.val_ann_path
        if hasattr(cfg, "test_evaluator"):
            cfg.test_evaluator.ann_file = user_config.data.val_ann_path
