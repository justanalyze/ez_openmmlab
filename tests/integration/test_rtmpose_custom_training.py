import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ez_openmmlab import RTMPose

def test_rtmpose_train_parameter_integration(tmp_path):
    """Verify that RTMPose.train correctly passes custom parameters to ConfigManager."""
    # 1. Setup mocks
    dataset_toml = tmp_path / "dataset.toml"
    dataset_toml.write_text("""
data_root = "."
dataset_name = "TestPose"
classes = ["person"]
[train]
ann_file = "t.json"
img_dir = "t/"
[val]
ann_file = "v.json"
img_dir = "v/"
""")

    mock_ds = MagicMock()
    mock_ds.dataset_name = "TestPose"
    mock_ds.classes = ["person"]
    mock_ds.data_root = str(tmp_path)
    mock_ds.train.ann_file = "t.json"
    mock_ds.train.img_dir = "t/"
    mock_ds.val.ann_file = "v.json"
    mock_ds.val.img_dir = "v/"
    mock_ds.test = None
    mock_ds.metainfo = {
        "keypoint_info": {"0": {}},
        "skeleton_info": {},
        "joint_weights": [],
        "sigmas": [],
    }

    # Mock the Runner and the base configuration loading to avoid full OpenMMLab init
    with patch("ez_openmmlab.core.engines.engine_base.Runner.from_cfg") as mock_runner_cls:
        mock_runner = MagicMock()
        mock_runner_cls.return_value = mock_runner
        
        with patch("ez_openmmlab.core.config_manager.DatasetConfig.from_toml", return_value=mock_ds):
            with patch("ez_openmmlab.core.engines.engine_base.EZMMLab._load_base_config") as mock_load:
                mock_cfg = MagicMock()
                mock_load.return_value = mock_cfg
                
                # 2. Instantiate and call train
                # We mock the resource resolution to avoid needing real config files
                with patch("ez_openmmlab.core.engines.engine_base.ensure_model_checkpoint", return_value=tmp_path / "m.pth"):
                    with patch("ez_openmmlab.core.engines.engine_base.get_config_file", return_value=tmp_path / "c.py"):
                        model = RTMPose("rtmpose_s")
                        
                        model.train(
                            dataset_config_path=dataset_toml,
                            input_size=(288, 384),
                            evaluator_metric="PCKAccuracy",
                            epochs=1,
                            batch_size=2
                        )

    # 3. Verify that the effective configuration reflects the overrides
    # The Effective Config is saved in work_dir/user_config.toml
    import tomli
    saved_config_path = Path("./runs/train/user_config.toml")
    with open(saved_config_path, "rb") as f:
        saved_data = tomli.load(f)
        
    assert saved_data["model"]["input_size"] == [288, 384]
    assert saved_data["training"]["evaluator_metric"] == "PCKAccuracy"
    # Sigma should be scaled (4.9 * 288/192 = 7.35, 5.66 * 384/256 = 8.49)
    assert saved_data["model"]["simcc_sigma"][0] == 7.35
    assert saved_data["model"]["simcc_sigma"][1] == 8.49
