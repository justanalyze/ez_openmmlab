from abc import ABC, abstractmethod
from typing import Any, Dict

from mmengine.config import Config

from ez_openmmlab.utils.toml_config import UserConfig


class BaseConfigSurgery(ABC):
    """Abstract base class for all 'surgery' operations on OpenMMLab configurations.

    'Surgery' refers to the automated patching and re-binding of internal MMEngine
    Config objects to ensure they reflect the user's TOML-first project definition.
    """

    @abstractmethod
    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        """Executes the specific configuration modification logic.

        Args:
            cfg: The MMEngine Config object to be modified.
            user_config: The high-level ez_openmmlab user configuration.
        """
        pass

    def _process_metainfo(self, metainfo: Dict[str, Any]) -> Dict[str, Any]:
        """Ensures that metadata keys (like keypoint IDs) are correctly typed for OpenMMLab.

        OpenMMLab expects certain metadata fields to use integer keys or specific structures.
        This utility handles the conversion from the flatter TOML-friendly formats.
        """
        processed = metainfo.copy()
        for key in ["keypoint_info", "skeleton_info"]:
            if key in processed and isinstance(processed[key], dict):
                processed[key] = {
                    int(k) if k.isdigit() else k: v for k, v in processed[key].items()
                }
        return processed