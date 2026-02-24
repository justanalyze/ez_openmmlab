import copy
from loguru import logger
from mmengine.config import Config

from ez_openmmlab.core.config_schema import UserConfig

from ..base import BaseConfigSurgery
from ..pipeline_patchers import PipelineTransformPatcherRegistry, _assign_cfg_value


class MMPoseInjector(BaseConfigSurgery):
    """Handles MMPose-specific configuration patches."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        """Patches model head, codec, pipelines, and test_cfg for MMPose models."""
        self._patch_head(cfg, user_config)
        self._patch_codec(cfg, user_config)
        self._patch_pipelines(cfg, user_config)
        self._patch_test_cfg(cfg, user_config)

    def _patch_head(self, cfg: Config, user_config: UserConfig) -> None:
        """Patches the model head with num_keypoints and resolution."""
        if not hasattr(cfg.model, "head"):
            return

        head = cfg.model.head
        target_num_keypoints = user_config.model.num_keypoints or user_config.model.num_classes

        if target_num_keypoints is not None:
            logger.info(f"[MMPoseInjector] Patching model head with {target_num_keypoints} keypoints")
            if hasattr(head, "out_channels"):
                head.out_channels = target_num_keypoints
            if hasattr(head, "num_keypoints"):
                head.num_keypoints = target_num_keypoints
                if hasattr(head, "head_module_cfg"):
                    head.head_module_cfg.num_classes = 1

        # Patch RTMPose head resolution
        input_size = getattr(user_config.model, "input_size", None)
        feature_map_size = getattr(user_config.model, "feature_map_size", None)

        if input_size:
            input_size_tuple = tuple(input_size) if isinstance(input_size, list) else input_size
            head.input_size = input_size_tuple

        if feature_map_size:
            head.in_featuremap_size = tuple(feature_map_size) if isinstance(feature_map_size, list) else feature_map_size

    def _patch_codec(self, cfg: Config, user_config: UserConfig) -> None:
        """Syncs the global codec with the new resolution and sigma."""
        if not hasattr(cfg, "codec"):
            return

        input_size = getattr(user_config.model, "input_size", None)
        simcc_sigma = getattr(user_config.model, "simcc_sigma", None)

        if not input_size:
            return

        if isinstance(input_size, list):
            input_size = tuple(input_size)
        if simcc_sigma and isinstance(simcc_sigma, list):
            simcc_sigma = tuple(simcc_sigma)

        logger.info(
            f"[MMPoseInjector] Patching global codec with input_size {input_size} "
            f"and sigma {simcc_sigma}"
        )
        cfg.codec.input_size = input_size
        if simcc_sigma:
            cfg.codec.sigma = simcc_sigma

    def _patch_pipelines(self, cfg: Config, user_config: UserConfig) -> None:
        """Updates pipeline transforms for scaling, rotation, and random flip."""
        # Consume from new augments section
        random_flip_prob = user_config.augments.random_flip_prob

        pipe_names = [
            "train_pipeline",
            "val_pipeline",
            "test_pipeline",
            "train_pipeline_stage1",
            "train_pipeline_stage2",
        ]

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

                # Apply registered patchers
                if transform_type:
                    patcher = PipelineTransformPatcherRegistry.get_patcher(
                        transform_type, "mmpose"
                    )
                    if patcher:
                        patcher.apply(transform_cfg, user_config, pipe_name)
                
                # Handle specific augmentation parameters
                if transform_type == "RandomFlip":
                    random_flip_exists = True
                    if random_flip_prob is not None:
                        _assign_cfg_value(transform_cfg, "prob", random_flip_prob)
                        logger.debug(
                            f"[MMPoseInjector] {pipe_name} -> RandomFlip: Set prob={random_flip_prob}"
                        )

            # Insert RandomFlip if not found and random_flip_prob is set
            if not random_flip_exists and random_flip_prob is not None and "train_pipeline" in pipe_name:
                insert_idx = -1
                for i, t in enumerate(pipeline):
                    if t.get("type") == "LoadImage":
                        insert_idx = i + 1
                        break
                if insert_idx == -1:
                    for i, t in enumerate(pipeline):
                        if t.get("type") == "PackPoseInputs":
                            insert_idx = i
                            break
                if insert_idx == -1:
                    insert_idx = len(pipeline)

                new_random_flip_cfg = dict(type="RandomFlip", direction="horizontal", prob=random_flip_prob)
                pipeline.insert(insert_idx, new_random_flip_cfg)
                logger.info(
                    f"[MMPoseInjector] Inserted RandomFlip (prob={random_flip_prob}) into {pipe_name}."
                )
            
            setattr(cfg, pipe_name, pipeline)


    def _patch_test_cfg(self, cfg: Config, user_config: UserConfig) -> None:
        """Patches model.test_cfg.input_size for models that define it."""
        input_size = getattr(user_config.model, "input_size", None)
        if not input_size:
            return

        if isinstance(input_size, list):
            input_size = tuple(input_size)

        if hasattr(cfg.model, "test_cfg") and "input_size" in cfg.model.test_cfg:
            cfg.model.test_cfg.input_size = input_size
            logger.debug(f"[MMPoseInjector] Patched model.test_cfg.input_size to {input_size}")