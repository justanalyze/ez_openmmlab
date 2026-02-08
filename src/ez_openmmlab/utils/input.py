from pathlib import Path
from typing import Union, List, Optional
from loguru import logger

def normalize_inputs(
    inputs: Union[str, Path, List[Union[str, Path]]]
) -> Union[str, List[str]]:
    """Normalizes input paths (single, list, or directory) into a format engines accept.

    Args:
        inputs: A single path, a list of paths, or a directory path.

    Returns:
        A list of image paths (strings) if the input was a list or directory,
        or a single string path if the input was a single file.
    """
    # Case 1: List of paths -> Convert all to strings
    if isinstance(inputs, list):
        return [str(p) for p in inputs]

    path_obj = Path(inputs)

    # Case 2: Directory -> Glob all images
    if path_obj.is_dir():
        extensions = {"*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"}
        images = []
        for ext in extensions:
            # Case insensitive globbing
            images.extend([str(p) for p in path_obj.glob(ext)])
            images.extend([str(p) for p in path_obj.glob(ext.upper())])

        if not images:
            logger.warning(f"No images found in directory: {inputs}")
        return sorted(list(set(images)))

    # Case 3: Single file path -> Return as string
    return str(path_obj)
