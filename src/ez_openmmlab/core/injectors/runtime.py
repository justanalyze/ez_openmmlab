from mmengine.config import Config

from ez_openmmlab.utils.toml_config import UserConfig

from .base import BaseConfigInjector


class RuntimeInjector(BaseConfigInjector):
    """Configures optimizer, AMP, epochs, and basic logging."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        training = user_config.training
        
        # Patch top-level variables for clarity in frozen config
        if hasattr(cfg, "base_lr"):
            cfg.base_lr = training.learning_rate
        if hasattr(cfg, "max_epochs"):
            cfg.max_epochs = training.epochs

        cfg.work_dir = training.work_dir
        cfg.train_cfg.max_epochs = training.epochs
        cfg.load_from = user_config.model.load_from
        cfg.log_level = training.log_level

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
