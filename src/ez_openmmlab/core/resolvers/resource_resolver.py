from pathlib import Path
from typing import Any, Dict, Optional, Union, TYPE_CHECKING

from loguru import logger
from pydantic import BaseModel, ConfigDict

if TYPE_CHECKING:
    from ez_openmmlab.core.config_manager import ConfigManager

from ez_openmmlab.core.schema.models import ModelName
from ez_openmmlab.core.utils.download import ensure_model_checkpoint


class ResolvedResources(BaseModel):
    """Encapsulates all resolved file paths and metadata for a model."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model_name: str
    checkpoint_path: Optional[Path]
    config_path: Path

    # Metadata from TOML if applicable
    num_classes: Optional[int] = None
    num_keypoints: Optional[int] = None
    metainfo: Optional[dict] = None
    architecture_params: Dict[str, Any] = {}

    # Contextual paths
    source_toml: Optional[Path] = None
    source_dir: Optional[Path] = None
    temp_config_file: Optional[Path] = None


class ResourceResolver:
    """Handles resolution of model names, checkpoints, and configurations."""

    def __init__(self, config_manager: "ConfigManager"):
        self._config_manager = config_manager

    def resolve(
        self,
        model: Union[ModelName, str, Path],
        checkpoint_path: Optional[Union[str, Path]] = None,
    ) -> ResolvedResources:
        """Resolves absolute paths for resources and initializes model state.

        Args:
            model: ModelName enum or path to config.toml.
            checkpoint_path: Path to model weights (.pth).

        Returns:
            A ResolvedResources object containing all necessary paths and metadata.
        """
        # Case 1: Custom Configuration via TOML
        if isinstance(model, (Path, str)) and str(model).endswith(".toml"):
            return self._resolve_from_toml(Path(model), checkpoint_path)

        # Case 2: Standard Model Name
        return self._resolve_from_name(model, checkpoint_path)

    def _resolve_from_toml(
        self, config_toml: Path, checkpoint_path: Optional[Union[str, Path]]
    ) -> ResolvedResources:
        """Handles resolution when a custom user_config.toml is provided."""
        source_toml = config_toml.absolute()
        source_dir = config_toml.parent.absolute()

        # 1. Resolve Weights
        final_checkpoint = None
        if checkpoint_path:
            final_checkpoint = Path(checkpoint_path).absolute()
        else:
            final_checkpoint = self._try_resolve_checkpoint(source_dir)

        # 2. Load metadata from TOML
        meta = self._config_manager.load_metadata_from_toml(config_toml)

        # 3. Generate temporary Python config
        temp_config = self._config_manager.prepare_config_file(
            config_toml, final_checkpoint
        )

        return ResolvedResources(
            model_name=meta.get("model_name", "custom_model"),
            checkpoint_path=final_checkpoint,
            config_path=temp_config,
            num_classes=meta.get("num_classes"),
            num_keypoints=meta.get("num_keypoints"),
            metainfo=meta.get("metainfo"),
            architecture_params=meta.get("architecture_params", {}),
            source_toml=source_toml,
            source_dir=source_dir,
            temp_config_file=temp_config,
        )

    def _resolve_from_name(
        self, model: Union[ModelName, str], checkpoint_path: Optional[Union[str, Path]]
    ) -> ResolvedResources:
        """Handles resolution for standard OpenMMLab model names."""
        from ez_openmmlab.core.config_manager import get_config_file

        model_str = model.value if isinstance(model, ModelName) else str(model)

        # ensure_model_checkpoint handles both finding existing and downloading
        resolved_checkpoint = ensure_model_checkpoint(model, checkpoint_path)
        resolved_config = get_config_file(model)

        return ResolvedResources(
            model_name=model_str,
            checkpoint_path=resolved_checkpoint,
            config_path=resolved_config,
        )

    def _try_resolve_checkpoint(self, directory: Path) -> Optional[Path]:
        """Smartly attempts to find a checkpoint in the given directory.

        Priority:
        1. best_*.pth
        2. Content of 'last_checkpoint' file
        """
        # 1. Search for 'best' checkpoint
        best_ckpts = list(directory.glob("best_*.pth"))
        if best_ckpts:
            # Pick the most recently modified 'best' checkpoint
            resolved = max(best_ckpts, key=lambda p: p.stat().st_mtime)
            logger.info(f"Smart-resolved 'best' checkpoint: {resolved.name}")
            return resolved

        # 2. Fallback to 'last_checkpoint' tracker
        last_ckpt_tracker = directory / "last_checkpoint"
        if last_ckpt_tracker.exists():
            try:
                ckpt_name = last_ckpt_tracker.read_text().strip()
                resolved = directory / ckpt_name
                if resolved.exists():
                    logger.info(
                        f"Smart-resolved last checkpoint from tracker: {resolved.name}"
                    )
                    return resolved
            except Exception as e:
                logger.warning(f"Failed to read 'last_checkpoint' tracker: {e}")

        return None
