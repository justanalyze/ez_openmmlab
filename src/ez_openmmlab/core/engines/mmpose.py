from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union

from loguru import logger
from mmengine.config import Config
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.core.inference.formatters import PoseResultFormatter
from ez_openmmlab.core.injectors.mmpose import MMPoseInjector
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils.context import switch_to_lib_root
from ez_openmmlab.utils.toml_config import (
    DataSection,
    ModelSection,
    TrainingSection,
    UserConfig,
)

from .engine_base import EZMMLab


class EZMMPose(EZMMLab):
    """Base engine for MMPose models."""

    def __init__(
        self,
        model: Union[ModelName, str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
        **kwargs,
    ):
        super().__init__(model, checkpoint_path, log_level, **kwargs)
        self._inferencer: Optional[MMPoseInferencer] = None
        self._formatter = PoseResultFormatter()

        # Pose-specific validation for custom TOML configs
        if isinstance(model, (Path, str)) and str(model).endswith(".toml"):
            if self.num_keypoints is None:
                raise ValueError(
                    f"Metadata 'num_keypoints' is missing in '{model}'. "
                    "Pose models require explicit 'num_keypoints' in the [model] section."
                )

    def _init_inferencer(self, device: str, **kwargs):
        """Lazy initialization of the MMPose inferencer with patching support."""
        if self._inferencer is None:
            logger.info(f"Initializing pose inferencer: {self.model}")
            pose_cfg = self._load_and_patch_config()

            # Delegate specific instantiation logic to children if needed,
            # but for most variants the standard MMPoseInferencer suffices.
            # We wrap this in switch_to_lib_root to ensure relative paths in the
            # config (like metainfo) are correctly resolved during model building.
            with switch_to_lib_root(self.model):
                self._inferencer = self._instantiate_inferencer(
                    pose_cfg, device, **kwargs
                )

    def _load_and_patch_config(self) -> Config:
        """Loads the pose config and applies runtime patches."""
        with switch_to_lib_root(self.model):
            cfg = Config.fromfile(str(self.config_path))

            # Only trigger patching if custom metadata is provided.
            if self.num_classes is not None or self.num_keypoints is not None:
                dummy_user_cfg = self._get_dummy_user_config()
                MMPoseInjector().apply(cfg, dummy_user_cfg)
            return cfg

    def _get_dummy_user_config(self) -> UserConfig:
        """Creates a dummy UserConfig to satisfy the injector interface."""
        return UserConfig(
            model=ModelSection(
                name=self.model,
                num_classes=self.num_classes,
                num_keypoints=self.num_keypoints,
            ),
            training=TrainingSection(num_workers=0, learning_rate=0.001),
            data=DataSection(root=""),
        )

    @abstractmethod
    def _instantiate_inferencer(
        self, cfg: Config, device: str, **kwargs
    ) -> MMPoseInferencer:
        """Instantiates the library-specific inferencer."""
        pass

    def _run_inference(
        self, inputs: list, out_dir: str, show: bool, **kwargs
    ) -> Union[dict, list]:
        """Calls the MMPoseInferencer and consumes the generator."""
        inferencer_kwargs = {
            "inputs": inputs,
            "show": show,
            "out_dir": out_dir if out_dir else None,
            **kwargs,
        }

        assert self._inferencer is not None, "Inferencer not initialized."

        # MMPoseInferencer is a generator. We must iterate/list it to trigger
        # the internal logic (inference, visualization, and saving).
        results_gen = self._inferencer(**inferencer_kwargs)
        return list(results_gen)
