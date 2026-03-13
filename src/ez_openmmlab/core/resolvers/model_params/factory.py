from typing import Optional

from .base import BaseModelParamsResolver
from .rtmpose import RTMPoseParamsResolver


class ModelParamsResolverFactory:
    """Factory to resolve the correct parameter resolver for a given model."""

    @staticmethod
    def get_resolver(model_name: str) -> Optional[BaseModelParamsResolver]:
        if "rtmpose" in model_name.lower():
            return RTMPoseParamsResolver()
        return None
