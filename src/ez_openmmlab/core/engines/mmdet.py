from pathlib import Path
from typing import Optional, Union

from loguru import logger
from mmdet.apis import DetInferencer
from mmdet.utils import register_all_modules

from ez_openmmlab.core.formatters import DetectionResultFormatter
from ez_openmmlab.schemas.model import ModelName

from .engine_base import EZMMLab

# Force registration of MMDet modules
register_all_modules(init_default_scope=True)


class EZMMDetector(EZMMLab):
    """Abstract base class for training and inference using MMDetection."""

    def __init__(
        self,
        model: Union[ModelName, str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
    ):
        super().__init__(model, checkpoint_path, log_level)
        self._inferencer: Optional[DetInferencer] = None
        self._formatter = DetectionResultFormatter()

    def _init_inferencer(self, device: str, **kwargs):
        """Lazy initialization of the DetInferencer."""
        if self._inferencer is None:
            logger.info(
                f"Initializing inferencer for model: {self.model} (using config: {self.config_path})"
            )
            self._inferencer = DetInferencer(
                model=str(self.config_path),
                weights=str(self.checkpoint_path),
                device=device,
            )

    def _run_inference(
        self, inputs: list, out_dir: str, show: bool, **kwargs
    ) -> Union[dict, list]:
        """Calls the DetInferencer with correct parameters."""
        # Map generic 'confidence' to mmdet 'pred_score_thr'
        confidence = kwargs.get("confidence") or kwargs.get("pred_score_thr", 0.3)

        assert self._inferencer is not None, "Inferencer not initialized."

        return self._inferencer(
            inputs, out_dir=out_dir, show=show, pred_score_thr=confidence
        )


