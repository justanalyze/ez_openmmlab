import copy
from loguru import logger
from mmengine.config import Config

from ez_openmmlab.utils.toml_config import UserConfig

from ..base import BaseConfigSurgery
from ..pipeline_patchers import PipelineTransformPatcherRegistry, _assign_cfg_value


class MMDetInjector(BaseConfigSurgery):
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
        """Updates pipeline transforms for scaling, rotation, and random flip."""
        input_size = getattr(user_config.model, "input_size", None)
        
        # Consume from new augments section
        random_flip_prob = user_config.augments.random_flip_prob
        scale_factor = user_config.augments.scale_factor

        pipe_names = [
            "train_pipeline",
            "val_pipeline",
            "test_pipeline",
            "train_pipeline_stage2",
        ]

        if "rtmdet" in user_config.model.name.value:
            pipe_names.append("tta_pipeline")

        for pipe_name in pipe_names:
            if not hasattr(cfg, pipe_name):
                continue

            pipeline = getattr(cfg, pipe_name)
            if not isinstance(pipeline, list):
                pipeline = pipeline.to_list() if isinstance(pipeline, Config) else []

            random_flip_exists = False
            
            for idx in range(len(pipeline)):
                transform_cfg = pipeline[idx]
                transform_type = transform_cfg.get("type")

                if transform_type:
                    patcher = PipelineTransformPatcherRegistry.get_patcher(
                        transform_type, "mmdet"
                    )
                    if patcher:
                        patcher.apply(transform_cfg, user_config, pipe_name)

                # Handle specific augmentation parameters
                if transform_type == "RandomFlip":
                    random_flip_exists = True
                    if random_flip_prob is not None:
                        _assign_cfg_value(transform_cfg, "prob", random_flip_prob)
                        logger.debug(
                            f"[MMDetInjector] {pipe_name} -> RandomFlip: Set prob={random_flip_prob}"
                        )
                elif transform_type == "RandomResize" and scale_factor is not None:
                    if isinstance(scale_factor, (list, tuple)):
                        _assign_cfg_value(transform_cfg, "ratio_range", scale_factor)
                        logger.debug(
                            f"[MMDetInjector] {pipe_name} -> RandomResize: Set ratio_range={scale_factor}"
                        )
                    elif isinstance(scale_factor, (int, float)):
                        # If a single value, use it as a fixed scale and clear ratio_range
                        _assign_cfg_value(transform_cfg, "scale", (input_size[0] * scale_factor, input_size[1] * scale_factor) if input_size else (int(640*scale_factor), int(640*scale_factor)))
                        transform_cfg.pop("ratio_range", None)
                        logger.debug(
                            f"[MMDetInjector] {pipe_name} -> RandomResize: Set fixed scale={scale_factor}"
                        )

            # Insert RandomFlip if not found and random_flip_prob is set
            if not random_flip_exists and random_flip_prob is not None and "train_pipeline" in pipe_name:
                insert_idx = -1
                for i, t in enumerate(pipeline):
                    if t.get("type") == "YOLOXHSVRandomAug":
                        insert_idx = i + 1
                        break
                if insert_idx == -1:
                    for i, t in enumerate(pipeline):
                        if t.get("type") == "PackDetInputs":
                            insert_idx = i
                            break
                if insert_idx == -1:
                    insert_idx = len(pipeline)

                new_random_flip_cfg = dict(type="RandomFlip", prob=random_flip_prob)
                pipeline.insert(insert_idx, new_random_flip_cfg)
                logger.info(
                    f"[MMDetInjector] Inserted RandomFlip (prob={random_flip_prob}) into {pipe_name}."
                )

            setattr(cfg, pipe_name, pipeline)
