from loguru import logger
from mmengine.config import Config
from ez_openmmlab.utils.toml_config import UserConfig
from .base import BaseConfigInjector


class MMPoseInjector(BaseConfigInjector):
    """Handles MMPose-specific configuration patches."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        """Patches the pose estimator head with the correct number of keypoints."""
        if not hasattr(cfg.model, "head"):
            return

        head = cfg.model.head
        # For pose, out_channels/num_keypoints is the target
        target = user_config.model.num_keypoints or user_config.model.num_classes

        if not target:
            return

        logger.info(f"[MMPoseInjector] Patching model head with {target} keypoints")

        # 1. RTMPose style
        if hasattr(head, "out_channels"):
            head.out_channels = target

        # 2. RTMO style
        if hasattr(head, "num_keypoints"):
            head.num_keypoints = target
            if hasattr(head, "head_module_cfg"):
                # RTMO head module also needs to know it's a single person class usually
                head.head_module_cfg.num_classes = 1
