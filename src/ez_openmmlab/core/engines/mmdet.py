from pathlib import Path
from typing import Any, Dict, Optional, Union

from loguru import logger
from mmengine.config import Config
from mmdet.apis import DetInferencer
from mmdet.utils import register_all_modules

from ez_openmmlab.core.inference.formatters import DetectionResultFormatter
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils.context import switch_to_lib_root

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
        **kwargs,
    ):
        super().__init__(model, checkpoint_path, log_level, **kwargs)
        self._inferencer: Optional[DetInferencer] = None
        self._formatter = DetectionResultFormatter()

    def _init_inferencer(self, device: str, **kwargs):
        """Lazy initialization of the DetInferencer with patching support."""
        if self._inferencer is None:
            logger.info(
                f"Initializing DetInferencer for model: {self.model} (using config: {self.config_path})"
            )
            det_cfg = self._load_and_patch_config(**kwargs)

            with switch_to_lib_root(self.model):
                self._inferencer = DetInferencer(
                    model=det_cfg,
                    weights=str(self.checkpoint_path),
                    device=device,
                )

    def _get_library_family(self) -> str:
        return "mmdet"

    def _run_inference(
        self, inputs: list, out_dir: str, show: bool, **kwargs
    ) -> Union[dict, list]:
        """Calls the DetInferencer and returns raw results."""
        # Map generic 'confidence' to mmdet 'pred_score_thr'
        confidence = kwargs.get("confidence") or kwargs.get("pred_score_thr", 0.3)

        assert self._inferencer is not None, "Inferencer not initialized."

        return self._inferencer(
            inputs, out_dir=out_dir, show=show, pred_score_thr=confidence
        )
