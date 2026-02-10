from abc import abstractmethod
from pathlib import Path
from typing import Optional, Union

from mmpose.apis import MMPoseInferencer

from ez_openmmlab.core.formatters import PoseResultFormatter
from ez_openmmlab.schemas.model import ModelName

from .engine_base import EZMMLab


class EZMMPose(EZMMLab):
    """Base engine for MMPose models.

    Provides shared utilities for interacting with MMPoseInferencer.
    """

    def __init__(
        self,
        model: Union[ModelName, str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
    ):
        super().__init__(model, checkpoint_path, log_level)
        self._inferencer: Optional[MMPoseInferencer] = None
        self._formatter = PoseResultFormatter()

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


    @abstractmethod
    def _init_inferencer(self, device: str, **kwargs):
        """Lazy initialization of the MMPose inferencer."""
        pass
