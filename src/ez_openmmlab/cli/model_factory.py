from pathlib import Path
from typing import Optional, Union

from ez_openmmlab.core.engines.engine_base import EZMMLab
from ez_openmmlab.core.schema.models import ModelName
from ez_openmmlab.models.mmdet import RTMDet
from ez_openmmlab.models.mmpose import RTMO, RTMPose


class ModelFactory:
    """Factory to instantiate models for the CLI."""

    @staticmethod
    def get_model(
        model_name: Union[ModelName, str],
        checkpoint_path: Optional[Union[str, Path]] = None,
        log_level: str = "INFO",
        **kwargs,
    ) -> EZMMLab:
        """Resolves and instantiates the correct model engine."""
        name_str = (
            model_name.value if isinstance(model_name, ModelName) else str(model_name)
        )
        # 1. Check for Pose models
        if "rtmpose" in name_str:
            return RTMPose(model_name, checkpoint_path, log_level, **kwargs)
        if "rtmo" in name_str:
            return RTMO(model_name, checkpoint_path, log_level, **kwargs)

        # 2. Default to RTMDet
        # (Could be expanded to other families as we scale)
        return RTMDet(model_name, checkpoint_path, log_level, **kwargs)
