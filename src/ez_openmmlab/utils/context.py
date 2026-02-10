import os
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


def get_project_root() -> Path:
    """Resolves the project root relative to this file."""
    return Path(__file__).resolve().parents[3]


@contextmanager
def switch_to_lib_root(model_name: str) -> Generator[Path, None, None]:
    """Context manager to temporarily switch to the appropriate library root.

    This is necessary because OpenMMLab configs often use relative paths
    that expect to be resolved from the library root (libs/mmdet or libs/mmpose).

    Args:
        model_name: The name of the model to determine the library root.

    Yields:
        The absolute path to the library root.
    """
    project_root = get_project_root()
    if "rtmpose" in model_name or "rtmo" in model_name:
        lib_root = project_root / "libs" / "mmpose"
    else:
        lib_root = project_root / "libs" / "mmdetection"

    old_cwd = os.getcwd()
    try:
        os.chdir(lib_root)
        yield lib_root
    finally:
        os.chdir(old_cwd)
