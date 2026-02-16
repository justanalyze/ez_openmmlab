from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from loguru import logger
from mmengine.config import Config

from ez_openmmlab.utils.toml_config import UserConfig


# --- Core Infrastructure ---

def _assign_cfg_value(cfg_obj: Union[Config, Dict[str, Any]], key: str, value: Any):
    """Assigns a value to a key in either a Config object or a dict."""
    if isinstance(cfg_obj, Config):
        # Use item access for Config objects to ensure persistence
        cfg_obj[key] = value
    elif isinstance(cfg_obj, dict):
        cfg_obj[key] = value
    else:
        logger.warning(f"Attempted to patch unsupported config object type: {type(cfg_obj)}")


class BasePipelineTransformPatcher(ABC):
    """Abstract base class for patching specific pipeline transforms."""

    def __init__(self, transform_type: str):
        self.transform_type = transform_type

    @abstractmethod
    def apply(
        self,
        transform_cfg: Union[Config, Dict[str, Any]],
        user_config: UserConfig,
        pipeline_name: str,
    ) -> None:
        """Applies the necessary patches to a single pipeline transform configuration."""
        pass

    def _get_input_size(self, user_config: UserConfig) -> Optional[Tuple[int, int]]:
        """Helper to safely extract input_size as a tuple from user config."""
        input_size = getattr(user_config.model, "input_size", None)
        if input_size:
            return tuple(input_size) if isinstance(input_size, list) else input_size
        return None

    def _log_patch(self, pipeline_name: str, key: str, value: Any):
        """Standardized logging for pipeline patches."""
        logger.debug(
            f"[PipelinePatcher] {pipeline_name} -> {self.transform_type}: "
            f"Set {key}={value}"
        )


class PipelineTransformPatcherRegistry:
    """A registry for pipeline transform patchers."""

    _patchers: Dict[str, Type[BasePipelineTransformPatcher]] = {}

    @classmethod
    def register(cls, patcher_cls: Type[BasePipelineTransformPatcher]):
        """Registers a new pipeline transform patcher."""
        if not issubclass(patcher_cls, BasePipelineTransformPatcher):
            raise TypeError(
                f"Patcher class {patcher_cls.__name__} must inherit from BasePipelineTransformPatcher"
            )
        
        instance = patcher_cls() 
        t_type = instance.transform_type
        
        if t_type in cls._patchers:
            logger.warning(f"Patcher for '{t_type}' already registered. Overwriting.")
            
        cls._patchers[t_type] = patcher_cls
        logger.debug(f"Registered pipeline patcher for '{t_type}'")

    @classmethod
    def get_patcher(cls, transform_type: str) -> Optional[BasePipelineTransformPatcher]:
        """Returns an instantiated patcher for the given transform type."""
        patcher_cls = cls._patchers.get(transform_type)
        return patcher_cls() if patcher_cls else None


# --- Generic Patcher Implementation ---

class InputSizePatcher(BasePipelineTransformPatcher):
    """A generic patcher that maps input_size to one or more config keys."""
    
    def __init__(self, transform_type: str, target_keys: List[str]):
        super().__init__(transform_type)
        self.target_keys = target_keys

    def apply(self, transform_cfg, user_config, pipeline_name):
        input_size = self._get_input_size(user_config)
        if not input_size:
            return
            
        for key in self.target_keys:
            # Only patch if the key exists or the transform is expected to have it
            if hasattr(transform_cfg, key) or key in transform_cfg:
                _assign_cfg_value(transform_cfg, key, input_size)
                self._log_patch(pipeline_name, key, input_size)


# --- Specialized Model Patchers ---

class RTMDetRandomResizePatcher(BasePipelineTransformPatcher):
    """Handles RTMDet's RandomResize which uses 2x input_size in Stage 1."""
    
    def __init__(self):
        super().__init__("RandomResize")

    def apply(self, transform_cfg, user_config, pipeline_name):
        input_size = self._get_input_size(user_config)
        if not input_size:
            return

        # RTMDet Stage 1 (train_pipeline) uses 2x scale for 'RandomResize'
        # Stage 2 (train_pipeline_stage2) uses 1x scale.
        ratio = 2.0 if pipeline_name == "train_pipeline" else 1.0
        scaled_size = tuple(int(x * ratio) for x in input_size)

        if hasattr(transform_cfg, "scale") or "scale" in transform_cfg:
            _assign_cfg_value(transform_cfg, "scale", scaled_size)
            self._log_patch(pipeline_name, "scale", scaled_size)


# --- Concrete Patcher Implementations ---

class TopdownAffinePatcher(InputSizePatcher):
    def __init__(self):
        super().__init__("TopdownAffine", ["input_size"])


class BottomupAffinePatcher(InputSizePatcher):
    def __init__(self):
        # BottomupRandomAffine uses both input_size and img_scale
        super().__init__("BottomupRandomAffine", ["input_size", "img_scale"])


class BottomupResizePatcher(InputSizePatcher):
    def __init__(self):
        super().__init__("BottomupResize", ["input_size"])


class MosaicPatcher(InputSizePatcher):
    def __init__(self):
        # Covers both Mosaic and CachedMosaic
        super().__init__("Mosaic", ["img_scale"])


class CachedMosaicPatcher(InputSizePatcher):
    def __init__(self):
        super().__init__("CachedMosaic", ["img_scale"])


class CachedMixUpPatcher(InputSizePatcher):
    def __init__(self):
        super().__init__("CachedMixUp", ["img_scale"])


class RandomCropPatcher(InputSizePatcher):
    def __init__(self):
        super().__init__("RandomCrop", ["crop_size"])


class PadPatcher(InputSizePatcher):
    def __init__(self):
        super().__init__("Pad", ["size"])


class ResizePatcher(InputSizePatcher):
    def __init__(self):
        # MMDetection/RTMDet standard Resize uses 'scale'
        super().__init__("Resize", ["scale"])


class TestTimeAugPatcher(BasePipelineTransformPatcher):
    """Handles multi-scale TTA patching for RTMDet.
    
    Adjusts nested Resize scales to [1.0x, 0.5x, 1.5x] and updates the Pad size
    to match the maximum scale (1.5x).
    """

    def __init__(self):
        super().__init__("TestTimeAug")

    def apply(self, transform_cfg, user_config, pipeline_name):
        input_size = self._get_input_size(user_config)
        if not input_size or "transforms" not in transform_cfg:
            return

        # RTMDet TTA Ratios: [1.0, 0.5, 1.5]
        s_10 = input_size
        s_05 = tuple(int(x * 0.5) for x in input_size)
        s_15 = tuple(int(x * 1.5) for x in input_size)
        scales = [s_10, s_05, s_15]

        # The 'transforms' key is a list of lists
        for sub_list in transform_cfg["transforms"]:
            if not isinstance(sub_list, (list, Config)):
                continue

            for t in sub_list:
                t_type = t.get("type")
                # 1. Update Resize scales
                if t_type == "Resize" and scales:
                    # RTMDet TTA typically has 3 Resizes in the first sub-list
                    # We assign the next scale in our sequence
                    new_scale = scales.pop(0)
                    _assign_cfg_value(t, "scale", new_scale)
                    self._log_patch(pipeline_name, "Resize.scale", new_scale)

                # 2. Update Pad size to match the largest scale (1.5x)
                elif t_type == "Pad":
                    _assign_cfg_value(t, "size", s_15)
                    self._log_patch(pipeline_name, "Pad.size", s_15)


# --- Registration ---

PipelineTransformPatcherRegistry.register(TopdownAffinePatcher)
PipelineTransformPatcherRegistry.register(BottomupAffinePatcher)
PipelineTransformPatcherRegistry.register(BottomupResizePatcher)
PipelineTransformPatcherRegistry.register(MosaicPatcher)
PipelineTransformPatcherRegistry.register(CachedMosaicPatcher)
PipelineTransformPatcherRegistry.register(CachedMixUpPatcher)
PipelineTransformPatcherRegistry.register(RandomCropPatcher)
PipelineTransformPatcherRegistry.register(PadPatcher)
PipelineTransformPatcherRegistry.register(ResizePatcher)
PipelineTransformPatcherRegistry.register(RTMDetRandomResizePatcher)
PipelineTransformPatcherRegistry.register(TestTimeAugPatcher)
