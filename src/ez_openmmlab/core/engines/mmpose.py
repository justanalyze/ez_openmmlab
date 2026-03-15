from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union

from mmengine.config import Config
from mmpose.apis import MMPoseInferencer

from ez_openmmlab.core.engines.engine_base import EZMMLab
from ez_openmmlab.core.inference.formatters import PoseResultFormatter
from ez_openmmlab.core.schema.models import ModelName
from ez_openmmlab.core.utils.context import switch_to_lib_root
from ez_openmmlab.core.validators import InputValidator


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

        # Pose-specific validation
        InputValidator.validate_mmpose_weights(
            model=model,
            using_custom_weights=self._using_custom_weights,
            num_keypoints=self.num_keypoints,
            num_classes=self.num_classes,
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
