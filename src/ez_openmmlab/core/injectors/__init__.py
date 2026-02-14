from typing import List

from .base import BaseConfigInjector
from .dataloader import DataloaderInjector
from .evaluator import EvaluatorInjector
from .mmdet import MMDetInjector
from .mmpose import MMPoseInjector
from .optimizer import OptimizerInjector
from .pipeline_patchers import PipelineTransformPatcherRegistry # Import the registry
from .runtime import RuntimeInjector


def get_injectors(model_name: str) -> List[BaseConfigInjector]:
    """Registry that returns the list of injectors required for a specific model."""
    # 1. Common injectors
    injectors: List[BaseConfigInjector] = [
        DataloaderInjector(),
        RuntimeInjector(),
        OptimizerInjector(),
        EvaluatorInjector(),
    ]

    # 2. Dynamic Library Selection
    if "rtmpose" in model_name or "rtmo" in model_name:
        injectors.append(MMPoseInjector())
    else:
        injectors.append(MMDetInjector())

    return injectors

# Ensure pipeline patchers are registered when this module is loaded
# This is handled within pipeline_patchers.py itself
