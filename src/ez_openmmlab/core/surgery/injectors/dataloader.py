from typing import Any, Dict

from mmengine.config import Config

from ez_openmmlab.core.config_schema import UserConfig

from ..base import BaseConfigSurgery


class DataloaderInjector(BaseConfigSurgery):
    """Configures dataset paths, batch sizes, and workers for all loaders."""

    def apply(self, cfg: Config, user_config: UserConfig) -> None:
        cfg.data_root = user_config.data.root

        # Sync dataset_type if we have a dynamic registration
        if user_config.data.registered_class_name:
            cfg.dataset_type = user_config.data.registered_class_name

        metainfo = self._assemble_metainfo(user_config)

        if metainfo:
            cfg.metainfo = metainfo

        for name in ["train_dataloader", "val_dataloader", "test_dataloader"]:
            if hasattr(cfg, name):
                self._configure_dataloader(
                    getattr(cfg, name), name, user_config, metainfo
                )

    def _assemble_metainfo(self, user_config: UserConfig) -> Dict[str, Any]:
        metainfo = {}
        if user_config.data.classes:
            metainfo["classes"] = user_config.data.classes
        if user_config.data.metainfo:
            metainfo.update(self._process_metainfo(user_config.data.metainfo))
        return metainfo

    def _configure_dataloader(
        self, dl: Config, name: str, user_config: UserConfig, metainfo: Dict[str, Any]
    ) -> None:
        dl.dataset.data_root = ""
        dl.batch_size = user_config.training.batch_size
        dl.num_workers = user_config.training.num_workers
        dl.persistent_workers = user_config.training.num_workers > 0

        # Use the dynamically registered class if available
        if user_config.data.registered_class_name:
            dl.dataset.type = user_config.data.registered_class_name

        if name == "train_dataloader":
            dl.dataset.ann_file = user_config.data.train_ann_path
            dl.dataset.data_prefix = {"img": user_config.data.train_img_path}
        else:
            dl.dataset.ann_file = user_config.data.val_ann_path
            dl.dataset.data_prefix = {"img": user_config.data.val_img_path}
