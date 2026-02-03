from typing import List
from .base import BaseConfigHandler
from .common import DataloaderHandler, RuntimeHandler
from .mmdet import MMDetHandler
from .mmpose import MMPoseHandler

def get_handlers(model_name: str) -> List[BaseConfigHandler]:
    """Registry that returns the list of handlers required for a specific model."""
    # 1. Common handlers
    handlers: List[BaseConfigHandler] = [
        DataloaderHandler(),
        RuntimeHandler(),
    ]

    # 2. Dynamic Library Selection
    if "rtmpose" in model_name or "rtmo" in model_name:
        handlers.append(MMPoseHandler())
    else:
        handlers.append(MMDetHandler())
    
    return handlers