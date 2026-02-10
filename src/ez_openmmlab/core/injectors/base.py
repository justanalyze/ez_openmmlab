from abc import ABC, abstractmethod
from typing import Any, Dict
from mmengine.config import Config
from ez_openmmlab.utils.toml_config import UserConfig


class BaseConfigInjector(ABC):
    """Abstract base class for all configuration plugins."""

    @abstractmethod
    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        """Applies specific configuration overrides to the OpenMMLab Config object."""
        pass

    def _process_metainfo(self, metainfo: Dict[str, Any]) -> Dict[str, Any]:
        """Shared utility to ensure metainfo keys are correctly typed."""
        processed = metainfo.copy()
        for key in ["keypoint_info", "skeleton_info"]:
            if key in processed and isinstance(processed[key], dict):
                processed[key] = {
                    int(k) if k.isdigit() else k: v for k, v in processed[key].items()
                }
        return processed
