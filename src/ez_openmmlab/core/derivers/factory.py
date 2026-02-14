from .base import BaseParameterDeriver
from .default import DefaultParameterDeriver
from .rtmpose import RTMPoseParameterDeriver


class DeriverFactory:
    """Factory to resolve the correct parameter deriver for a given model."""

    @staticmethod
    def get_deriver(model_name: str) -> BaseParameterDeriver:
        if "rtmpose" in model_name.lower():
            return RTMPoseParameterDeriver()
        return DefaultParameterDeriver()
