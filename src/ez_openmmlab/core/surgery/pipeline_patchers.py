from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from loguru import logger
from mmengine.config import Config

from ez_openmmlab.core.schema.config import UserConfig


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
    """A registry for pipeline transform patchers, partitioned by model family.
    
    This ensures that family-specific transforms (like RTMDet's RandomResize)
    don't conflict or leak into other model families.
    """

    # Structured as { family_name: { transform_type: patcher_class } }
    _patchers: Dict[str, Dict[str, Type[BasePipelineTransformPatcher]]] = {
        "common": {},
        "mmdet": {},
        "mmpose": {},
    }

    # Map user-facing keys to the internal transform types they affect.
    # This is used for validation and documentation.
    _user_keys: Dict[str, List[str]] = {
        "common": ["random_flip_prob"],
        "mmdet": ["scale_factor"],
        "mmpose": ["scale_factor", "rotate_factor"],
    }

    @classmethod
    def get_supported_augments(cls, family: str) -> List[str]:
        """Returns the list of user-facing augmentation keys supported for a family."""
        keys = set(cls._user_keys.get("common", []))
        keys.update(cls._user_keys.get(family, []))
        return sorted(list(keys))

    @classmethod
    def register(cls, family: str, patcher_cls: Type[BasePipelineTransformPatcher]):
        """Registers a new pipeline transform patcher for a specific family.
        
        Args:
            family: One of 'common', 'mmdet', or 'mmpose'.
            patcher_cls: The patcher class to register.
        """
        if family not in cls._patchers:
            raise ValueError(f"Invalid family '{family}' for pipeline patcher registration.")

        if not issubclass(patcher_cls, BasePipelineTransformPatcher):
            raise TypeError(
                f"Patcher class {patcher_cls.__name__} must inherit from BasePipelineTransformPatcher"
            )
        
        # Instantiate once to get the transform_type
        instance = patcher_cls() 
        t_type = instance.transform_type
        
        if t_type in cls._patchers[family]:
            logger.warning(f"Patcher for '{t_type}' already registered in family '{family}'. Overwriting.")
            
        cls._patchers[family][t_type] = patcher_cls
        logger.debug(f"Registered {family} pipeline patcher for '{t_type}'")

    @classmethod
    def get_patcher(cls, transform_type: str, family: str) -> Optional[BasePipelineTransformPatcher]:
        """Returns an instantiated patcher for the given transform type and family.
        
        Priority:
        1. Family-specific patcher (e.g. 'mmdet')
        2. Common patcher
        """
        # 1. Try family-specific
        patcher_cls = cls._patchers.get(family, {}).get(transform_type)
        
        # 2. Fallback to common
        if not patcher_cls:
            patcher_cls = cls._patchers["common"].get(transform_type)

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


class BottomupRandomAffinePatcher(InputSizePatcher):
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


class YOLOXMixUpPatcher(InputSizePatcher):
    def __init__(self):
        super().__init__("YOLOXMixUp", ["img_scale"])


class RandomCropPatcher(InputSizePatcher):
    def __init__(self):
        super().__init__("RandomCrop", ["crop_size"])


class RandomBBoxTransformPatcher(BasePipelineTransformPatcher):
    def __init__(self):
        super().__init__("RandomBBoxTransform")

    def apply(self, transform_cfg, user_config, pipeline_name):
        # Pull from the NEW augments section
        scale_factor = user_config.augments.scale_factor
        rotate_factor = user_config.augments.rotate_factor

        if scale_factor is not None:
            # RandomBBoxTransform uses scale_factor as a list/tuple range
            if isinstance(scale_factor, (float, int)):
                scale_factor = [scale_factor, scale_factor] # Ensure it's a range
            _assign_cfg_value(transform_cfg, "scale_factor", scale_factor)
            self._log_patch(pipeline_name, "scale_factor", scale_factor)
        if rotate_factor is not None:
            _assign_cfg_value(transform_cfg, "rotate_factor", rotate_factor)
            self._log_patch(pipeline_name, "rotate_factor", rotate_factor)


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

# Common (Applicable to multiple families if types match)
PipelineTransformPatcherRegistry.register("common", PadPatcher)

# MMDet specific
PipelineTransformPatcherRegistry.register("mmdet", ResizePatcher)
PipelineTransformPatcherRegistry.register("mmdet", RTMDetRandomResizePatcher)
PipelineTransformPatcherRegistry.register("mmdet", MosaicPatcher)
PipelineTransformPatcherRegistry.register("mmdet", CachedMosaicPatcher)
PipelineTransformPatcherRegistry.register("mmdet", CachedMixUpPatcher)
PipelineTransformPatcherRegistry.register("mmdet", RandomCropPatcher) # Confirmed mmdet only
PipelineTransformPatcherRegistry.register("mmdet", TestTimeAugPatcher)

# MMPose specific
PipelineTransformPatcherRegistry.register("mmpose", TopdownAffinePatcher)
PipelineTransformPatcherRegistry.register("mmpose", BottomupRandomAffinePatcher)
PipelineTransformPatcherRegistry.register("mmpose", BottomupResizePatcher)
PipelineTransformPatcherRegistry.register("mmpose", MosaicPatcher) # Used by RTMO
PipelineTransformPatcherRegistry.register("mmpose", YOLOXMixUpPatcher) # Used by RTMO
PipelineTransformPatcherRegistry.register("mmpose", RandomBBoxTransformPatcher) # Used by RTMPose
