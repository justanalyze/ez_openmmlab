from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union

from loguru import logger
from mmengine.config import Config
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.core.engines.engine_base import EZMMLab
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


class EZMMPose(EZMMLab):
    """Base class for all MMPose-based engines (RTMPose, RTMO, etc.)."""

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

    def _validate_inputs(
        self,
        model: Union[ModelName, str, Path],
        checkpoint_path: Optional[Union[str, Path]],
    ) -> None:
        """Pose-specific input validation."""
        super()._validate_inputs(model, checkpoint_path)

        # For pose models, we often need to know the number of keypoints
        # if using custom weights without a full config.toml
        if self._using_custom_weights and not str(model).endswith(".toml"):
            if self.num_keypoints is None and self.num_classes is None:
                raise ValueError(
                    "MMPose models require 'num_keypoints' or 'num_classes' to be specified "
                    "when loading custom weights without a config.toml."
                )

    def _init_inferencer(self, device: str, **kwargs):
        """Lazy initialization of the MMPose inferencer with patching support."""
        if self._inferencer is None:
            pose_cfg = self._load_and_patch_config(**kwargs)

            # Delegate specific instantiation logic to children if needed,
            # but for most variants the standard MMPoseInferencer suffices.
            # We wrap this in switch_to_lib_root to ensure relative paths in the
            # config (like metainfo) are correctly resolved during model building.
            with switch_to_lib_root(self.model):
                self._inferencer = self._instantiate_inferencer(
                    pose_cfg, device, **kwargs
                )

    def _load_and_patch_config(self, **kwargs) -> Config:
        """Loads the pose config and applies runtime patches."""
        with switch_to_lib_root(self.model):
            cfg = Config.fromfile(str(self.config_path))

            # Trigger patching if custom metadata or architecture params are provided
            dummy_user_cfg = self._get_dummy_user_config(**kwargs)
            MMPoseInjector().apply(cfg, dummy_user_cfg)

            return cfg

    def _get_dummy_user_config(self, **kwargs) -> UserConfig:
        """Creates a dummy UserConfig to satisfy the injector interface.

        This ensures that architecture-specific parameters passed during predict()
        or loaded from config.toml are correctly picked up by the MMPoseInjector.
        """
        model_params = {
            "name": self.model,
            "num_classes": self.num_classes or 1,  # Default to 1 class for pose
            "num_keypoints": self.num_keypoints,
        }
        # 1. Use stored architecture_params (from config.toml)
        model_params.update(self.architecture_params)

        # 2. Inject architecture-specific parameters passed to predict()
        model_params.update(kwargs)

        return UserConfig(
            model=ModelSection(**model_params),
            training=TrainingSection(num_workers=0, learning_rate=0.001),
            data=DataSection(root=""),
        )

    def _get_library_family(self) -> str:
        return "mmpose"

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
