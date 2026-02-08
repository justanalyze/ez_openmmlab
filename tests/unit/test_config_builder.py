import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ez_openmmlab.core.config_builder import UserConfigBuilder
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils.toml_config import UserConfig

class TestUserConfigBuilder:
    def test_build_from_model_name(self, tmp_path):
        builder = UserConfigBuilder()
        model_name = ModelName.RTM_DET_TINY
        checkpoint = Path("ckpt.pth")
        
        # Mock dataset.toml loading
        dummy_dataset_path = tmp_path / "dataset.toml"
        dummy_dataset_path.touch()
        
        with patch("ez_openmmlab.core.config_builder.DatasetConfig.from_toml") as mock_ds_load:
            mock_ds = MagicMock()
            mock_ds.classes = ["cat", "dog"]
            mock_ds.data_root = Path("/data")
            mock_ds.train.ann_file = "train.json"
            mock_ds.train.img_dir = "train/"
            mock_ds.val.ann_file = "val.json"
            mock_ds.val.img_dir = "val/"
            mock_ds.test = None
            mock_ds.metainfo = None
            mock_ds_load.return_value = mock_ds
            
            user_config = builder.build(
                model=model_name,
                dataset_config_path=dummy_dataset_path,
                checkpoint_path=checkpoint,
                epochs=50,
                batch_size=4
            )
            
            assert isinstance(user_config, UserConfig)
            assert user_config.model.name == model_name
            assert user_config.model.num_classes == 2
            assert user_config.model.load_from == str(checkpoint)
            assert user_config.training.epochs == 50
            assert user_config.training.batch_size == 4

    def test_auto_load_metadata_from_config(self, tmp_path):
        """Test metadata auto-loading from a nearby config file."""
        builder = UserConfigBuilder()
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()
        checkpoint = checkpoint_dir / "best.pth"
        
        # Create a dummy user_config.toml next to checkpoint
        config_path = checkpoint_dir / "user_config.toml"
        config_path.write_text("""
        [model]
        name = "rtmdet_tiny"
        num_classes = 5
        
        [data]
        root = "/data"
        train_ann = "a"
        train_img = "b"
        val_ann = "c"
        val_img = "d"
        
        [training]
        epochs = 1
        batch_size = 1
        """)
        
        metadata = builder.load_metadata_from_checkpoint(checkpoint)
        assert metadata["num_classes"] == 5
        assert metadata["num_keypoints"] is None

    def test_resolve_temp_config_from_toml(self, tmp_path):
        """Test that passing a toml path generates a temp python config."""
        builder = UserConfigBuilder()
        config_toml = tmp_path / "custom_config.toml"
        config_toml.write_text("""
        [model]
        name = "rtmdet_tiny"
        num_classes = 3
        
        [data]
        root = "/d"
        train_ann = "a"
        train_img = "b"
        val_ann = "c"
        val_img = "d"
        
        [training]
        epochs = 1
        batch_size = 1
        """)
        
        checkpoint = Path("ckpt.pth")
        
        with patch("ez_openmmlab.core.config_builder.get_config_file") as mock_get_cfg:
            mock_get_cfg.return_value = Path("/libs/mmdet/configs/base.py")
            
            # This logic mimics what will be moved from EZMMLab._resolve_model_config
            temp_config = builder.prepare_config_file(config_toml, checkpoint)
            
            try:
                assert temp_config.exists()
                assert temp_config.suffix == ".py"
                content = temp_config.read_text()
                assert '_base_ = ["/libs/mmdet/configs/base.py"]' in content
            finally:
                if temp_config.exists():
                    temp_config.unlink()

    def test_cleanup_temp_config(self, tmp_path):
        builder = UserConfigBuilder()
        temp_file = tmp_path / "temp.py"
        temp_file.touch()
        assert temp_file.exists()
        
        builder.cleanup_temp_config(temp_file)
        assert not temp_file.exists()

