import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ez_openmmlab.engines.mmdet import EZMMDetector
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils.toml_config import load_user_config

class ConcreteEZDetector(EZMMDetector):
    def _init_inferencer(self, device):
        pass

@patch("ez_openmmlab.core.base.Runner")
@patch("ez_openmmlab.core.base.ensure_model_checkpoint")
@patch("ez_openmmlab.schemas.dataset.DatasetConfig.from_toml")
def test_train_saves_base_config_path(mock_ds_from_toml, mock_ensure, mock_runner, tmp_path):
    """Test that training saves the absolute path to the base python config."""
    work_dir = tmp_path / "runs" / "train"
    dataset_toml = tmp_path / "dataset.toml"
    dataset_toml.touch()
    
    # Mock dataset config
    mock_ds = MagicMock()
    mock_ds.classes = ["cat"]
    mock_ds.data_root = "/data"
    mock_ds.train.ann_file = "t.json"
    mock_ds.train.img_dir = "t/"
    mock_ds.val.ann_file = "v.json"
    mock_ds.val.img_dir = "v/"
    mock_ds.test = None
    mock_ds.metainfo = None
    mock_ds_from_toml.return_value = mock_ds
    
    mock_ensure.return_value = Path("dummy.pth")
    
    detector = ConcreteEZDetector(model=ModelName.RTM_DET_TINY)
    
    # Mock _apply_common_overrides to avoid needing a real config object
    with patch.object(detector, "_apply_common_overrides"):
        detector.train(
            dataset_config_path=dataset_toml,
            work_dir=str(work_dir),
            epochs=1
        )
    
    saved_config_path = work_dir / "user_config.toml"
    assert saved_config_path.exists()
    
    user_cfg = load_user_config(saved_config_path)
    assert user_cfg.model.base_config_path is not None
    assert Path(user_cfg.model.base_config_path).is_absolute()
    assert "rtmdet_tiny" in user_cfg.model.base_config_path
