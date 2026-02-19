from loguru import logger
from mmengine.config import Config

from ez_openmmlab.utils.toml_config import UserConfig

from ..base import BaseConfigSurgery


class OptimizerInjector(BaseConfigSurgery):
    """Applies global optimization hyperparameters (weight decay, etc.) to the config."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        weight_decay = user_config.training.weight_decay
        if weight_decay is None:
            return

        if not hasattr(cfg, "optim_wrapper") or not hasattr(
            cfg.optim_wrapper, "optimizer"
        ):
            return

        cfg.optim_wrapper.optimizer.weight_decay = weight_decay
        logger.debug(f"[OptimizerInjector] Set weight_decay to {weight_decay}")
