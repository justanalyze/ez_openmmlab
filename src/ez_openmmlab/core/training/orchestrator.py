from pathlib import Path
from typing import Any, Dict, TYPE_CHECKING

from loguru import logger
from mmengine.config import Config
from mmengine.runner import Runner

from ez_openmmlab.core.datasets import DynamicDatasetRegistry
from ez_openmmlab.core.schema.config import UserConfig, save_user_config
from ez_openmmlab.core.surgery import get_surgeries
from ez_openmmlab.core.config_manager import get_config_file

if TYPE_CHECKING:
    from ez_openmmlab.core.engines.engine_base import EZMMLab


class TrainingOrchestrator:
    """Manages the full MMEngine training lifecycle for an engine."""

    def __init__(self):
        pass

    def run(self, engine: "EZMMLab", config: UserConfig, dry_run: bool = False) -> None:
        """Orchestrates the internal OpenMMLab setup and execution."""
        # 1. Register Dataset
        DynamicDatasetRegistry.register_dataset(config, engine._get_library_family())

        # 2. Setup Artifacts
        work_dir = Path(config.training.work_dir)
        work_dir.mkdir(parents=True, exist_ok=True)
        config.model.base_config_path = str(
            get_config_file(config.model.name).absolute()
        )

        save_user_config(config, work_dir / "user_config.toml")

        # 3. Prepare Final Configuration
        # We use ConfigManager for loading and patching
        cfg = engine._config_manager.load_base_config(
            config.model.name.value, Path(config.model.base_config_path)
        )
        engine._config_manager.patch_config(cfg, engine.model, config)

        final_config_path = (
            work_dir / f"{config.model.name.value}_{config.data.dataset_name}.py"
        )
        engine._config_manager.dump_config(cfg, final_config_path)

        if dry_run:
            logger.info(f"Dry run: {final_config_path}")
            return

        # 4. Execute Training
        self._execute_runner(final_config_path, config.training.log_level)

        # 5. Synchronize State back to the engine
        self._update_engine_state(engine, work_dir, final_config_path)

    def _execute_runner(self, config_path: Path, log_level: str) -> None:
        """Initializes the MMEngine Runner and starts training."""
        logger.info(f"Starting MMEngine Runner: {config_path}")
        runner = Runner.from_cfg(Config.fromfile(str(config_path)))
        runner.train()

    def _update_engine_state(self, engine: "EZMMLab", work_dir: Path, config_path: Path) -> None:
        """Synchronizes engine state with newly trained weights and persistent configs."""
        logger.info("Synchronizing model state after training...")
        
        # Use the resolver to find the new checkpoint
        new_checkpoint = engine._resource_resolver._try_resolve_checkpoint(work_dir)
        if new_checkpoint:
            engine.checkpoint_path = new_checkpoint
            engine._using_custom_weights = True

        engine._source_toml = (work_dir / "user_config.toml").absolute()
        engine._source_dir = work_dir.absolute()
        engine.config_path = config_path

        # Refresh metadata
        meta = engine._config_manager.load_metadata_from_toml(engine._source_toml)
        engine.num_classes = meta.get("num_classes")
        engine.num_keypoints = meta.get("num_keypoints")
        engine.metainfo = meta.get("metainfo")
        engine.architecture_params = meta.get("architecture_params", {})

        # Clear inferencer to force re-initialization on next predict() call
        if hasattr(engine, "_inferencer"):
            engine._inferencer = None
