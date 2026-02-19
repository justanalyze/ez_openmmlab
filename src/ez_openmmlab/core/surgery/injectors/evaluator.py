from loguru import logger
from mmengine.config import Config

from ez_openmmlab.utils.toml_config import UserConfig

from ..base import BaseConfigSurgery


class EvaluatorInjector(BaseConfigSurgery):
    """Configures the metric(s) for validation and testing across all model families."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        metrics = user_config.training.evaluator_metric
        if not metrics:
            return

        # Convert input to a uniform list of (string or dict) for processing
        if isinstance(metrics, (str, dict)):
            metric_list = [metrics]
        else:
            metric_list = metrics

        for eval_name in ["val_evaluator", "test_evaluator"]:
            if not hasattr(cfg, eval_name):
                continue

            # We prioritize the absolute path from user_config.data
            ann_file = user_config.data.val_ann_path

            # Rebuild the evaluator config as a list of dicts (MMEngine multi-metric style)
            evaluator_cfg = []
            for m in metric_list:
                if isinstance(m, str):
                    # Standard string metric
                    evaluator_cfg.append(dict(type=m, ann_file=ann_file))
                elif isinstance(m, dict):
                    # Advanced dictionary metric (e.g. PCKAccuracy with threshold)
                    # We create a copy and ensure ann_file is present if not already specified
                    m_copy = m.copy()
                    if "ann_file" not in m_copy:
                        m_copy["ann_file"] = ann_file
                    evaluator_cfg.append(m_copy)

            # If only one metric, keep it as a dict for simplicity/compatibility
            if len(evaluator_cfg) == 1:
                setattr(cfg, eval_name, evaluator_cfg[0])
            else:
                setattr(cfg, eval_name, evaluator_cfg)

            logger.info(f"[EvaluatorInjector] Configured {eval_name} with {metric_list}")
