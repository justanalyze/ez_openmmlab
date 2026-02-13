import os
import platform
from pathlib import Path
from typing import Union


def get_user_cache_dir() -> Path:
    """Returns the platform-specific cache directory for ez_openmmlab.

    - Windows: %LOCALAPPDATA%/ez_openmmlab/Cache
    - macOS: ~/Library/Caches/ez_openmmlab
    - Linux/Other: ~/.cache/ez_openmmlab
    """
    system = platform.system()
    if system == "Windows":
        # Use LOCALAPPDATA if available, fallback to ~/.cache
        base = os.environ.get("LOCALAPPDATA")
        if base and os.path.isdir(base):
            return Path(base) / "ez_openmmlab" / "Cache"
        return Path.home() / ".cache" / "ez_openmmlab"
    elif system == "Darwin":  # macOS
        return Path.home() / "Library" / "Caches" / "ez_openmmlab"
    else:  # Linux and others (standard XDG convention)
        return Path.home() / ".cache" / "ez_openmmlab"


def get_unique_dir(base_dir: Union[str, Path]) -> Path:
    """Returns a unique directory path by appending a numbered suffix if it already exists.

    Example:
        If 'runs/preds' exists, returns 'runs/preds_1'
        If 'runs/preds_1' exists, returns 'runs/preds_2'

    Args:
        base_dir: The desired base directory path.

    Returns:
        A Path object representing a unique directory.
    """
    path = Path(base_dir)
    if not path.exists():
        return path

    parent = path.parent
    stem = path.name

    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}"
        if not new_path.exists():
            return new_path
        counter += 1
