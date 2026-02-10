import warnings

# Suppress noisy library warnings immediately on import
warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated")
warnings.filterwarnings("ignore", message="A new version of Albumentations is available")

from mmdet.utils import register_all_modules as register_mmdet

try:
    from mmpose.utils import register_all_modules as register_mmpose

    register_mmpose(init_default_scope=False)
except ImportError:
    pass

# Ensure MMDet modules are registered with default scope immediately
register_mmdet(init_default_scope=True)

from .core.engines.engine_base import EZMMLab
from .core.engines.mmdet import EZMMDetector
from .core.engines.mmpose import EZMMPose
from .models.mmdet import RTMDet
from .models.mmpose import RTMO, RTMPose

__all__ = [
    "RTMDet",
    "RTMPose",
    "RTMO",
    "EZMMLab",
    "EZMMDetector",
    "EZMMPose",
    "mute_warnings",
]


def mute_warnings():
    """Helper to suppress MMLab and noisy library verbosity manually."""
    import logging
    import warnings

    from mmengine.logging import MMLogger

    # Suppress noisy MMLab and PyTorch warnings
    warnings.filterwarnings("ignore", message=".*LocalVisBackend")
    warnings.filterwarnings("ignore", message=".*meshgrid")
    warnings.filterwarnings("ignore", message=".*bbox is out of bounds")
    warnings.filterwarnings("ignore", message=".*polygon is out of bounds")
    warnings.filterwarnings(
        "ignore", message="A new version of Albumentations is available"
    )

    # Suppress mmengine log warnings
    logging.getLogger("mmengine").setLevel(logging.ERROR)
    try:
        MMLogger.get_instance("mmengine").setLevel(logging.ERROR)
    except Exception:
        pass
