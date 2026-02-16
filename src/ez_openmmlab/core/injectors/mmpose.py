from loguru import logger
from mmengine.config import Config

from ez_openmmlab.utils.toml_config import UserConfig

from .base import BaseConfigInjector
from .pipeline_patchers import PipelineTransformPatcherRegistry


class MMPoseInjector(BaseConfigInjector):
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
        """Updates input_size in all pipeline transforms using the registry."""
        for pipe_name in [
            "train_pipeline",
            "val_pipeline",
            "test_pipeline",
            "train_pipeline_stage1",
            "train_pipeline_stage2",
        ]:
            if not hasattr(cfg, pipe_name):
                continue
            
            pipeline_data = getattr(cfg, pipe_name).to_list() if isinstance(getattr(cfg, pipe_name), Config) else list(getattr(cfg, pipe_name))

            for transform_cfg in pipeline_data:
                transform_type = transform_cfg.get("type")
                if transform_type:
                    patcher = PipelineTransformPatcherRegistry.get_patcher(transform_type)
                    if patcher:
                        patcher.apply(transform_cfg, user_config, pipe_name)
            
            setattr(cfg, pipe_name, pipeline_data)

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