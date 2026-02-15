from typing import List

from .base import BaseConfigInjector
from .dataloader import DataloaderInjector
from .evaluator import EvaluatorInjector
from .mmdet import MMDetInjector
from .mmpose import MMPoseInjector
from .optimizer import OptimizerInjector
from .pipeline_patchers import PipelineTransformPatcherRegistry # Import the registry
from .runtime import RuntimeInjector
from .structural import DataloaderRebinder, HookRebinder # Import the new rebinders


def get_injectors(model_name: str) -> List[BaseConfigInjector]:
    """Registry that returns the list of injectors required for a specific model."""
    # 1. Common injectors (Patching values)
    injectors: List[BaseConfigInjector] = [
        DataloaderInjector(),
        RuntimeInjector(),
        OptimizerInjector(),
        EvaluatorInjector(),
    ]

    # 2. Model-Family Specific (Patching values)
    if "rtmpose" in model_name or "rtmo" in model_name:
        injectors.append(MMPoseInjector())
    else:
        injectors.append(MMDetInjector())

    # 3. Structural Re-binding (Wiring the config together)
    # These MUST run last to ensure they bind to the final patched values.
    injectors.extend([
        DataloaderRebinder(),
        HookRebinder(),
    ])

    return injectors

# Ensure pipeline patchers are registered when this module is loaded
# This is handled within pipeline_patchers.py itself