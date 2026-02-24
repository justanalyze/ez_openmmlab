from typing import Optional

from loguru import logger
from mmengine.config import Config

from ez_openmmlab.core.config_schema import UserConfig

from .base import BaseConfigSurgery


class StructuralRebinder(BaseConfigSurgery):
    """Base class for re-binding internal configuration references."""

    def _get_path(self, obj: any, path: str) -> Optional[any]:
        """Safely navigates a path like 'dataloader.dataset.pipeline'."""
        parts = path.split(".")
        current = obj
        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        return current

    def _set_path(self, obj: any, path: str, value: any) -> bool:
        """Safely sets a value at a nested path."""
        parts = path.split(".")
        target_attr = parts[-1]
        parent_path = ".".join(parts[:-1])

        parent = self._get_path(obj, parent_path) if parent_path else obj
        if parent is None:
            return False

        if hasattr(parent, target_attr) or (
            isinstance(parent, dict) and target_attr in parent
        ):
            if hasattr(parent, "__setitem__") and not hasattr(
                parent, target_attr
            ):
                parent[target_attr] = value
            else:
                setattr(parent, target_attr, value)
            return True
        return False


class DataloaderRebinder(StructuralRebinder):
    """Ensures dataloaders point to the correctly patched pipeline variables."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        # Standard mappings for OpenMMLab families
        mappings = [
            ("train_dataloader.dataset.pipeline", "train_pipeline"),
            ("val_dataloader.dataset.pipeline", "val_pipeline"),
            ("test_dataloader.dataset.pipeline", "test_pipeline"),
        ]

        # Fallback for detection where val/test often share 'test_pipeline'
        if not hasattr(cfg, "val_pipeline") and hasattr(cfg, "test_pipeline"):
            mappings.append(
                ("val_dataloader.dataset.pipeline", "test_pipeline")
            )

        for target_path, source_var in mappings:
            if hasattr(cfg, source_var):
                source_val = getattr(cfg, source_var)
                if self._set_path(cfg, target_path, source_val):
                    logger.debug(
                        f"[Structural] Re-bound {target_path} to {source_var}"
                    )


class HookRebinder(StructuralRebinder):
    """Ensures custom hooks use the patched pipeline variables (e.g., Stage 2 transitions)."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        if not hasattr(cfg, "custom_hooks"):
            return

        new_max_epochs = user_config.training.epochs

        for hook in cfg.custom_hooks:
            hook_type = hook.get("type", "")
            if hook_type == "PipelineSwitchHook" or hook_type.endswith(
                ".PipelineSwitchHook"
            ):
                # 1. Sync Pipeline Reference
                if hasattr(cfg, "train_pipeline_stage2"):
                    hook.switch_pipeline = cfg.train_pipeline_stage2
                    logger.debug(
                        f"[Structural] Re-bound {hook_type} to train_pipeline_stage2"
                    )

                # 2. Sync Switch Timing
                stage2_duration = getattr(cfg, "stage2_num_epochs", 20)
                new_switch_epoch = max(1, new_max_epochs - stage2_duration)

                hook.switch_epoch = new_switch_epoch
                logger.info(
                    f"[Structural] {hook_type} synchronized: "
                    f"Stage 2 will start at epoch {new_switch_epoch} (Duration: {stage2_duration})"
                )


class CodecRebinder(StructuralRebinder):
    """Ensures head decoder and pipeline encoder point to the patched global codec."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        if not hasattr(cfg, "codec"):
            return

        codec = cfg.codec

        # 1. Sync Model Head Decoder
        if self._set_path(cfg, "model.head.decoder", codec):
            logger.debug("[Structural] Re-bound model.head.decoder to codec")

        # 2. Sync Pipeline Encoders (Search through all standard pipelines)
        for pipe_name in [
            "train_pipeline",
            "val_pipeline",
            "test_pipeline",
            "train_pipeline_stage1",
            "train_pipeline_stage2",
        ]:
            if not hasattr(cfg, pipe_name):
                continue

            pipeline = getattr(cfg, pipe_name)
            for transform in pipeline:
                if transform.get("type") == "GenerateTarget":
                    transform.encoder = codec
                    logger.debug(
                        f"[Structural] Re-bound {pipe_name} GenerateTarget.encoder to codec"
                    )
