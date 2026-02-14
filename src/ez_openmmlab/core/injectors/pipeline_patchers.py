from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from loguru import logger
from mmengine.config import Config

from ez_openmmlab.utils.toml_config import UserConfig


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
        
        # Instantiate with default to get the intended transform_type
        instance = patcher_cls() 
        t_type = instance.transform_type
        
        if not t_type:
            raise ValueError(f"Patcher class {patcher_cls.__name__} did not define a transform_type")

        if t_type in cls._patchers:
            logger.warning(
                f"PipelineTransformPatcher for '{t_type}' already registered. Overwriting."
            )
        cls._patchers[t_type] = patcher_cls
        logger.debug(
            f"Registered pipeline transform patcher for '{t_type}'"
        )

    @classmethod
    def get_patcher(
        cls, transform_type: str
    ) -> Optional[BasePipelineTransformPatcher]:
        """Returns an instantiated patcher for the given transform type."""
        patcher_cls = cls._patchers.get(transform_type)
        if patcher_cls:
            return patcher_cls() 
        return None

# --- Helper for dynamic assignment ---
def _assign_cfg_value(cfg_obj: Union[Config, Dict[str, Any]], key: str, value: Any):
    """Assigns a value to a key in either a Config object or a dict."""
    if isinstance(cfg_obj, Config):
        # Use item access for Config objects to ensure persistence in some MMEngine versions
        cfg_obj[key] = value
    elif isinstance(cfg_obj, dict):
        cfg_obj[key] = value
    else:
        logger.warning(f"Attempted to patch unsupported config object type: {type(cfg_obj)}")


# --- Concrete Patcher Implementations ---

class TopdownAffinePatcher(BasePipelineTransformPatcher):
    def __init__(self, transform_type: str = "TopdownAffine"):
        super().__init__(transform_type)

    def apply(
        self,
        transform_cfg: Union[Config, Dict[str, Any]],
        user_config: UserConfig,
        pipeline_name: str,
    ) -> None:
        input_size = getattr(user_config.model, "input_size", None)
        if input_size:
            input_size_tuple = tuple(input_size) if isinstance(input_size, list) else input_size
            _assign_cfg_value(transform_cfg, "input_size", input_size_tuple)
            logger.debug(
                f"[MMPoseInjector] Updated {pipeline_name} {self.transform_type} input_size to {input_size}"
            )


class BottomupAffinePatcher(BasePipelineTransformPatcher):
    def __init__(self, transform_type: str = "BottomupRandomAffine"):
        super().__init__(transform_type)

    def apply(
        self,
        transform_cfg: Union[Config, Dict[str, Any]],
        user_config: UserConfig,
        pipeline_name: str,
    ) -> None:
        input_size = getattr(user_config.model, "input_size", None)
        if input_size:
            input_size_tuple = tuple(input_size) if isinstance(input_size, list) else input_size
            if hasattr(transform_cfg, "input_size") or "input_size" in transform_cfg:
                _assign_cfg_value(transform_cfg, "input_size", input_size_tuple)
            if hasattr(transform_cfg, "img_scale") or "img_scale" in transform_cfg:
                _assign_cfg_value(transform_cfg, "img_scale", input_size_tuple)
            logger.debug(
                f"[MMPoseInjector] Updated {pipeline_name} {self.transform_type} input_size/img_scale to {input_size}"
            )

class BottomupResizePatcher(BasePipelineTransformPatcher):
    def __init__(self, transform_type: str = "BottomupResize"):
        super().__init__(transform_type)

    def apply(
        self,
        transform_cfg: Union[Config, Dict[str, Any]],
        user_config: UserConfig,
        pipeline_name: str,
    ) -> None:
        input_size = getattr(user_config.model, "input_size", None)
        if input_size:
            input_size_tuple = tuple(input_size) if isinstance(input_size, list) else input_size
            _assign_cfg_value(transform_cfg, "input_size", input_size_tuple)
            logger.debug(
                f"[MMPoseInjector] Updated {pipeline_name} {self.transform_type} input_size to {input_size}"
            )

class MosaicPatcher(BasePipelineTransformPatcher):

    def __init__(self, transform_type: str = "Mosaic"):

        super().__init__(transform_type)



    def apply(

        self,

        transform_cfg: Union[Config, Dict[str, Any]],

        user_config: UserConfig,

        pipeline_name: str,

    ) -> None:

        input_size = getattr(user_config.model, "input_size", None)

        if input_size:

            input_size_tuple = tuple(input_size) if isinstance(input_size, list) else input_size

            _assign_cfg_value(transform_cfg, "img_scale", input_size_tuple)

            logger.debug(

                f"[MMPoseInjector] Updated {pipeline_name} {self.transform_type} img_scale to {input_size}"

            )





class ResizePatcher(BasePipelineTransformPatcher):

    def __init__(self, transform_type: str = "Resize"):

        super().__init__(transform_type)



    def apply(

        self,

        transform_cfg: Union[Config, Dict[str, Any]],

        user_config: UserConfig,

        pipeline_name: str,

    ) -> None:

        input_size = getattr(user_config.model, "input_size", None)

        if input_size:

            input_size_tuple = (

                tuple(input_size) if isinstance(input_size, list) else input_size

            )

            # MMDet Resize uses 'scale'

            _assign_cfg_value(transform_cfg, "scale", input_size_tuple)

            logger.debug(

                f"[MMPoseInjector] Updated {pipeline_name} {self.transform_type} scale to {input_size}"

            )





# Register the patchers

PipelineTransformPatcherRegistry.register(TopdownAffinePatcher)

PipelineTransformPatcherRegistry.register(BottomupAffinePatcher)

PipelineTransformPatcherRegistry.register(BottomupResizePatcher)

PipelineTransformPatcherRegistry.register(MosaicPatcher)

PipelineTransformPatcherRegistry.register(ResizePatcher)
