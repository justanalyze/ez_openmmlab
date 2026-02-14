from loguru import logger
from mmengine.config import Config

from ez_openmmlab.utils.toml_config import UserConfig

from .base import BaseConfigInjector


class EvaluatorInjector(BaseConfigInjector):
    """Configures the metric(s) for validation and testing across all model families."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        metrics = user_config.training.evaluator_metric
        if not metrics:
            return

        # Convert single string to list for uniform handling
        metric_list = [metrics] if isinstance(metrics, str) else metrics

        for eval_name in ["val_evaluator", "test_evaluator"]:
            if not hasattr(cfg, eval_name):
                continue

            current_eval = getattr(cfg, eval_name)
            # Maintain the original ann_file path if possible, fallback to config paths
            # Note: 'val_ann' is the common field in DataSection
            ann_file = (
                current_eval.get("ann_file")
                if isinstance(current_eval, dict)
                else user_config.data.val_ann
            )

            # Rebuild the evaluator config as a list of dicts (MMEngine multi-metric style)
            evaluator_cfg = [dict(type=m, ann_file=ann_file) for m in metric_list]

            # If only one metric, keep it as a dict for simplicity/compatibility
            if len(evaluator_cfg) == 1:
                setattr(cfg, eval_name, evaluator_cfg[0])
            else:
                setattr(cfg, eval_name, evaluator_cfg)

            logger.info(f"[EvaluatorInjector] Configured {eval_name} with {metric_list}")
