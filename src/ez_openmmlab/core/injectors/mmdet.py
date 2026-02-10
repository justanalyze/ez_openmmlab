from loguru import logger
from mmengine.config import Config

from ez_openmmlab.utils.toml_config import UserConfig

from .base import BaseConfigInjector


class MMDetInjector(BaseConfigInjector):
    """Handles MMDetection-specific configuration patches."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        """Patches the detector head with the correct number of classes."""
        if not hasattr(cfg.model, "bbox_head"):
            return

        model_cfg = user_config.model
        num_classes = model_cfg.num_classes

        logger.info(
            f"[MMDetInjector] Patching model.bbox_head.num_classes to {num_classes}"
        )

        bbox_head = cfg.model.bbox_head
        if isinstance(bbox_head, list):
            for head in bbox_head:
                head.num_classes = num_classes
        else:
            bbox_head.num_classes = num_classes
