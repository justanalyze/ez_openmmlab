import os
import importlib.util
from contextlib import contextmanager
from pathlib import Path
from typing import Generator


def get_project_root() -> Path:
    """Resolves the project root relative to this file."""
    return Path(__file__).resolve().parents[4]


@contextmanager
def switch_to_lib_root(model_name: str) -> Generator[Path, None, None]:
    """Context manager to temporarily switch to the appropriate library root.

    This is necessary because OpenMMLab configs often use relative paths
    that expect to be resolved from the library root (mmdet or mmpose).

    Args:
        model_name: The name of the model to determine the library root.

    Yields:
        The absolute path to the library root.
    """
    pkg_name = "mmpose" if ("rtmpose" in model_name or "rtmo" in model_name) else "mmdet"
    spec = importlib.util.find_spec(pkg_name)
    if not spec or not spec.origin:
        raise ImportError(f"Could not find installed package: {pkg_name}")

    lib_root = Path(spec.origin).parent

    old_cwd = os.getcwd()
    try:
        os.chdir(lib_root)
        yield lib_root
    finally:
        os.chdir(old_cwd)
