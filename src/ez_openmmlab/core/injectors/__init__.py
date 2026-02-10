from typing import List

from .base import BaseConfigInjector
from .common import DataloaderInjector, RuntimeInjector
from .mmdet import MMDetInjector
from .mmpose import MMPoseInjector


def get_injectors(model_name: str) -> List[BaseConfigInjector]:
    """Registry that returns the list of injectors required for a specific model."""
    # 1. Common injectors
    injectors: List[BaseConfigInjector] = [
        DataloaderInjector(),
        RuntimeInjector(),
    ]

    # 2. Dynamic Library Selection
    if "rtmpose" in model_name or "rtmo" in model_name:
        injectors.append(MMPoseInjector())
    else:
        injectors.append(MMDetInjector())

    return injectors
