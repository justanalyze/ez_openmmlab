from loguru import logger
from mmengine.config import Config

from ez_openmmlab.utils.toml_config import UserConfig

from .base import BaseConfigInjector


class MMPoseInjector(BaseConfigInjector):
    """Handles MMPose-specific configuration patches."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        """Patches model head, codec, pipelines, and optimizer for RTMPose."""
        self._patch_head(cfg, user_config)
        self._patch_codec(cfg, user_config)
        self._patch_pipelines(cfg, user_config)
        self._patch_optimizer(cfg, user_config)
        self._patch_evaluators(cfg, user_config)

    def _patch_head(self, cfg: Config, user_config: UserConfig) -> None:
        if not hasattr(cfg.model, "head"):
            return

        head = cfg.model.head
        target = user_config.model.num_keypoints or user_config.model.num_classes

        if not target:
            return

        logger.info(f"[MMPoseInjector] Patching model head with {target} keypoints")

        if hasattr(head, "out_channels"):
            head.out_channels = target

        if hasattr(head, "num_keypoints"):
            head.num_keypoints = target
            if hasattr(head, "head_module_cfg"):
                head.head_module_cfg.num_classes = 1

        # Patch RTMPose head input/feature map size
        if user_config.model.input_size:
            head.input_size = user_config.model.input_size
        if user_config.model.feature_map_size:
            head.in_featuremap_size = user_config.model.feature_map_size

    def _patch_codec(self, cfg: Config, user_config: UserConfig) -> None:
        """Syncs the SimCC codec with the new resolution and sigma."""
        if not hasattr(cfg, "codec"):
            return

        logger.info(
            f"[MMPoseInjector] Patching codec with size {user_config.model.input_size} "
            f"and sigma {user_config.model.simcc_sigma}"
        )
        cfg.codec.input_size = user_config.model.input_size
        cfg.codec.sigma = user_config.model.simcc_sigma

    def _patch_pipelines(self, cfg: Config, user_config: UserConfig) -> None:
        """Updates input_size in all pipeline steps (TopdownAffine)."""
        input_size = user_config.model.input_size
        
        for pipe_name in ["train_pipeline", "val_pipeline", "test_pipeline"]:
            if not hasattr(cfg, pipe_name):
                continue
            
            pipeline = getattr(cfg, pipe_name)
            for transform in pipeline:
                if transform.get("type") == "TopdownAffine":
                    transform["input_size"] = input_size
                    logger.debug(f"[MMPoseInjector] Updated {pipe_name} TopdownAffine to {input_size}")

    def _patch_optimizer(self, cfg: Config, user_config: UserConfig) -> None:
        """Applies weight decay to the optimizer."""
        if not hasattr(cfg, "optim_wrapper") or not hasattr(cfg.optim_wrapper, "optimizer"):
            return
        
        cfg.optim_wrapper.optimizer.weight_decay = user_config.training.weight_decay
        logger.debug(f"[MMPoseInjector] Set weight_decay to {user_config.training.weight_decay}")

    def _patch_evaluators(self, cfg: Config, user_config: UserConfig) -> None:
        """Configures the metric(s) for validation and testing."""
        metrics = user_config.training.evaluator_metric
        if not metrics:
            return

        # Convert single string to list for uniform handling
        metric_list = [metrics] if isinstance(metrics, str) else metrics

        for eval_name in ["val_evaluator", "test_evaluator"]:
            if not hasattr(cfg, eval_name):
                continue
            
            # Simple case: Replace existing single metric or update list
            evaluator = getattr(cfg, eval_name)
            if isinstance(evaluator, dict):
                # If it's a standard OpenMMLab evaluator dict
                if len(metric_list) == 1:
                    evaluator["type"] = metric_list[0]
                else:
                    # Multi-metric support usually requires a list of dicts in MMEngine
                    # but for now we'll support the primary type override.
                    evaluator["type"] = metric_list[0] 
            
            logger.info(f"[MMPoseInjector] Configured {eval_name} with {metric_list[0]}")
