from loguru import logger
from pathlib import Path
from typing import Optional, Union

from ez_openmmlab.core.engines.mmdet import EZMMDetector
from ez_openmmlab.schemas.model import ModelName

class RTMDet(EZMMDetector):
    """RTMDet implementation for fast object detection and instance segmentation.

    Args:
        model: Name of the RTMDet variant (e.g., 'rtmdet_tiny', 'rtmdet-ins_s') OR path to a config.toml.
        checkpoint_path: Path to a custom .pth file. If None, it will use the official pretrained weights.
        log_level: Global logging level for the internal engine ('INFO', 'WARNING', 'ERROR').
    """

    def __init__(
        self,
        model: Union[ModelName, str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
        **kwargs,
    ):
        super().__init__(model, checkpoint_path, log_level, **kwargs)