from typing import List

from .base import BaseConfigSurgery
from .injectors import (
    DataloaderInjector,
    EvaluatorInjector,
    MMDetInjector,
    MMPoseInjector,
    OptimizerInjector,
    RuntimeInjector,
)
from .rebinders import CodecRebinder, DataloaderRebinder, HookRebinder


def get_surgeries(model_name: str) -> List[BaseConfigSurgery]:
    """Registry that returns the list of configuration surgeries required for a specific model."""
    # 1. Common surgeries
    surgeries: List[BaseConfigSurgery] = [
        DataloaderInjector(),
        RuntimeInjector(),
        OptimizerInjector(),
        EvaluatorInjector(),
    ]

    # 2. Model-Family Specific surgeries
    if "rtmpose" in model_name or "rtmo" in model_name:
        surgeries.append(MMPoseInjector())
        surgeries.append(CodecRebinder())  # MMPose uses Codecs
    else:
        surgeries.append(MMDetInjector())

    # 3. Structural Re-binding (Wiring the config together)
    # These MUST run last to ensure they bind to the final patched values.
    surgeries.extend(
        [
            DataloaderRebinder(),
            HookRebinder(),
        ]
    )

    return surgeries

