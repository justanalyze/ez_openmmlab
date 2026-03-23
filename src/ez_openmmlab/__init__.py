# Check dependencies before importing anything else

from ez_openmmlab._check_deps import check_dependencies

check_dependencies()


def mute_warnings():
    """Helper to suppress MMLab and noisy library verbosity manually."""
    import logging
    import sys
    import warnings

    from loguru import logger
    from mmengine.logging import MMLogger

    # 0. Globally configure Loguru to INFO level by default
    logger.remove()
    logger.add(sys.stderr, level="INFO")

    # 1. Suppress standard Python warnings
    warnings.filterwarnings("ignore", message=".*pkg_resources is deprecated")
    warnings.filterwarnings("ignore", message=".*LocalVisBackend")
    warnings.filterwarnings("ignore", message=".*meshgrid")
    warnings.filterwarnings("ignore", message=".*bbox is out of bounds")
    warnings.filterwarnings("ignore", message=".*polygon is out of bounds")
    warnings.filterwarnings(
        "ignore", message="A new version of Albumentations is available"
    )

    # 2. Suppress standard logging for noisy libraries
    noisy_loggers = [
        "mmengine",
        "PIL",
        "matplotlib",
        "urllib3",
        "albumentations",
    ]
    for name in noisy_loggers:
        logging.getLogger(name).setLevel(logging.ERROR)

    # 3. Suppress MMEngine specific logger
    try:
        MMLogger.get_instance("mmengine").setLevel(logging.ERROR)
    except Exception:
        pass


# Apply immediate silencing on import to ensure 'EZ' first impression
mute_warnings()

from mmdet.utils import register_all_modules as register_mmdet

try:
    from mmpose.utils import register_all_modules as register_mmpose

    register_mmpose(init_default_scope=False)
except ImportError:
    pass

# Ensure MMDet modules are registered with default scope immediately
register_mmdet(init_default_scope=True)

from .models.mmdet import RTMDet
from .models.mmpose import RTMO, RTMPose

__version__ = "0.1.2"
__all__ = [
    "RTMDet",
    "RTMPose",
    "RTMO",
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
