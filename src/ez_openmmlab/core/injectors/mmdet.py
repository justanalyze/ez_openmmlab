from loguru import logger
from mmengine.config import Config

from ez_openmmlab.utils.toml_config import UserConfig

from .base import BaseConfigInjector
from .pipeline_patchers import PipelineTransformPatcherRegistry


class MMDetInjector(BaseConfigInjector):
    """Handles MMDetection-specific configuration patches."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        """Patches model head and pipelines for RTMDet."""
        self._patch_head(cfg, user_config)
        self._patch_pipelines(cfg, user_config)

    def _patch_head(self, cfg: Config, user_config: UserConfig) -> None:
        if not hasattr(cfg.model, "bbox_head"):
            return

        model_cfg = user_config.model
        num_classes = model_cfg.num_classes

        # Only patch if explicitly provided
        if num_classes is None:
            return

        logger.info(
            f"[MMDetInjector] Patching model.bbox_head.num_classes to {num_classes}"
        )

        # RTMDet style
        bbox_head = cfg.model.bbox_head
        if isinstance(bbox_head, (list, tuple)):
            for head in bbox_head:
                head.num_classes = num_classes
        else:
            bbox_head.num_classes = num_classes

    def _patch_pipelines(self, cfg: Config, user_config: UserConfig) -> None:
        """Updates input_size in all pipeline steps (Resize)."""
        input_size = getattr(user_config.model, "input_size", None)
        if not input_size:
            return

        for pipe_name in ["train_pipeline", "val_pipeline", "test_pipeline"]:
            if not hasattr(cfg, pipe_name):
                continue

            pipeline = getattr(cfg, pipe_name)
            # Iterate through indices to ensure modifications persist in the Config object
            for idx in range(len(pipeline)):
                transform_cfg = pipeline[idx]
                transform_type = transform_cfg.get("type")

                if transform_type:
                    patcher = PipelineTransformPatcherRegistry.get_patcher(
                        transform_type
                    )
                    if patcher:
                        patcher.apply(pipeline[idx], user_config, pipe_name)