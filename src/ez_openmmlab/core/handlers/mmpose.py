from loguru import logger
from mmengine.config import Config
from ez_openmmlab.utils.toml_config import UserConfig
from .base import BaseConfigHandler

class MMPoseHandler(BaseConfigHandler):
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

        logger.info(f"[MMPoseHandler] Patching model head with {target} keypoints")

        # 1. RTMPose style
        if hasattr(head, "out_channels"):
            head.out_channels = target
            
        # 2. RTMO style
        if hasattr(head, "num_keypoints"):
            head.num_keypoints = target
            if hasattr(head, "head_module_cfg"):
                # RTMO head module also needs to know it's a single person class usually
                head.head_module_cfg.num_classes = 1

        # 3. Patch dataset_meta and metainfo to avoid TTA errors (flip_indices shape mismatch)
        # We must reset flip_indices because the default ones won't match our new count.
        # We use a neutral flip (every point flips to itself) as a safe default.
        flip_indices = list(range(target))
        
        if hasattr(cfg, "dataset_meta") and cfg.dataset_meta:
            logger.debug(f"[MMPoseHandler] Patching cfg.dataset_meta.flip_indices (count: {target})")
            cfg.dataset_meta.flip_indices = flip_indices
            
        if hasattr(cfg, "metainfo") and cfg.metainfo:
            logger.debug(f"[MMPoseHandler] Patching cfg.metainfo.flip_indices (count: {target})")
            cfg.metainfo.flip_indices = flip_indices

        # 4. Patch individual dataloaders to ensure TTA/Pipeline consistency
        for name in ["train_dataloader", "val_dataloader", "test_dataloader"]:
            if hasattr(cfg, name):
                dl = getattr(cfg, name)
                if hasattr(dl.dataset, "metainfo") and dl.dataset.metainfo:
                    dl.dataset.metainfo.flip_indices = flip_indices

        # 5. Disable flip_test as a last resort if TTA is still failing
        if hasattr(cfg.model, "test_cfg") and cfg.model.test_cfg:
            logger.debug("[MMPoseHandler] Disabling flip_test to avoid flip_indices mismatch")
            cfg.model.test_cfg.flip_test = False
