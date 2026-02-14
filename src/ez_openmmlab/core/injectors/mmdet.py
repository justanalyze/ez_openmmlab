from loguru import logger
from mmengine.config import Config

from ez_openmmlab.utils.toml_config import UserConfig

from .base import BaseConfigInjector


class MMDetInjector(BaseConfigInjector):
    """Handles MMDetection-specific configuration patches."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        """Patches model head, pipelines, and optimizer for RTMDet."""
        self._patch_head(cfg, user_config)
        self._patch_pipelines(cfg, user_config)
        self._patch_optimizer(cfg, user_config)
        self._patch_evaluators(cfg, user_config)

    def _patch_head(self, cfg: Config, user_config: UserConfig) -> None:
        if not hasattr(cfg.model, "bbox_head"):
            return

        model_cfg = user_config.model
        num_classes = model_cfg.num_classes

        if num_classes is None:
            return

        logger.info(
            f"[MMDetInjector] Patching model.bbox_head.num_classes to {num_classes}"
        )

        # RTMDet style
        bbox_head = cfg.model.bbox_head
        if isinstance(bbox_head, list):
            for head in bbox_head:
                head.num_classes = num_classes
        else:
            bbox_head.num_classes = num_classes

    def _patch_pipelines(self, cfg: Config, user_config: UserConfig) -> None:
        """Updates input_size in all pipeline steps (Resize)."""
        input_size = getattr(user_config.model, "input_size", None)
        if not input_size:
            return

        # MMDet uses 'scale' in Resize transforms
        # Usually (width, height)
        for pipe_name in ["train_pipeline", "val_pipeline", "test_pipeline"]:
            if not hasattr(cfg, pipe_name):
                continue

            pipeline = getattr(cfg, pipe_name)
            for transform in pipeline:
                if transform.get("type") == "Resize":
                    transform["scale"] = input_size
                    logger.debug(
                        f"[MMDetInjector] Updated {pipe_name} Resize scale to {input_size}"
                    )

    def _patch_optimizer(self, cfg: Config, user_config: UserConfig) -> None:
        """Applies weight decay to the optimizer."""
        if not hasattr(cfg, "optim_wrapper") or not hasattr(
            cfg.optim_wrapper, "optimizer"
        ):
            return

        cfg.optim_wrapper.optimizer.weight_decay = user_config.training.weight_decay
        logger.debug(
            f"[MMDetInjector] Set weight_decay to {user_config.training.weight_decay}"
        )

    def _patch_evaluators(self, cfg: Config, user_config: UserConfig) -> None:
        """Configures the metric(s) for validation and testing."""
        metrics = user_config.training.evaluator_metric
        if not metrics:
            return

        metric_list = [metrics] if isinstance(metrics, str) else metrics

        for eval_name in ["val_evaluator", "test_evaluator"]:
            if not hasattr(cfg, eval_name):
                continue

            current_eval = getattr(cfg, eval_name)
            ann_file = (
                current_eval.get("ann_file")
                if isinstance(current_eval, dict)
                else user_config.data.val_ann
            )

            evaluator_cfg = [dict(type=m, ann_file=ann_file) for m in metric_list]

            if len(evaluator_cfg) == 1:
                setattr(cfg, eval_name, evaluator_cfg[0])
            else:
                setattr(cfg, eval_name, evaluator_cfg)

            logger.info(f"[MMDetInjector] Configured {eval_name} with {metric_list}")