import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import cv2

from ez_openmmlab import RTMDet
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils.toml_config import load_user_config

@patch("ez_openmmlab.core.base.Runner")
@patch("ez_openmmlab.core.base.ensure_model_checkpoint")
@patch("ez_openmmlab.schemas.dataset.DatasetConfig.from_toml")
@patch("ez_openmmlab.engines.mmdet.DetInferencer")
@patch("cv2.imread")
def test_full_custom_config_flow(mock_imread, mock_inferencer_cls, mock_ds_from_toml, mock_ensure, mock_runner, tmp_path):
    """
    Integration test for the full flow:
    1. Train a model (mocked runner) -> saves user_config.toml with base_config_path.
    2. Initialize a new model using that saved config.toml.
    3. Run inference (mocked inferencer).
    """
    work_dir = tmp_path / "runs" / "train"
    dataset_toml = tmp_path / "dataset.toml"
    dataset_toml.touch()
    
    # Mock image
    mock_imread.return_value = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # 1. Mock dataset config
    mock_ds = MagicMock()
    mock_ds.classes = ["cat", "dog"]
    mock_ds.data_root = "/data"
    mock_ds.train.ann_file = "t.json"
    mock_ds.train.img_dir = "t/"
    mock_ds.val.ann_file = "v.json"
    mock_ds.val.img_dir = "v/"
    mock_ds.test = None
    mock_ds.metainfo = None
    mock_ds_from_toml.return_value = mock_ds
    
    mock_ensure.return_value = Path("base_weights.pth")
    
    # --- STEP 1: TRAIN ---
    detector = RTMDet(model=ModelName.RTM_DET_TINY)
    
    with patch.object(detector, "_apply_common_overrides"):
        detector.train(
            dataset_config_path=dataset_toml,
            work_dir=str(work_dir),
            epochs=1
        )
    
    saved_config_path = work_dir / "user_config.toml"
    assert saved_config_path.exists()
    
    # --- STEP 2: LOAD FROM CUSTOM CONFIG ---
    custom_checkpoint = work_dir / "best.pth"
    custom_checkpoint.touch()
    
    # Re-mock ensure_model_checkpoint for the new instance
    # In a real scenario, this would just return the path as it's already local
    with patch("ez_openmmlab.core.base.ensure_model_checkpoint", return_value=custom_checkpoint):
        custom_detector = RTMDet(model=saved_config_path, checkpoint_path=custom_checkpoint)
        
        assert custom_detector.model == ModelName.RTM_DET_TINY.value
        assert custom_detector.num_classes == 2
        assert custom_detector.config_path == custom_detector._temp_config_file
        
        # --- STEP 3: PREDICT ---
        mock_inferencer_instance = MagicMock()
        mock_inferencer_instance.return_value = {"predictions": [{"labels": [0], "scores": [0.9], "bboxes": [[10, 10, 50, 50]]}]}
        mock_inferencer_cls.return_value = mock_inferencer_instance
        
        results = custom_detector.predict("dummy.jpg")
        
        assert len(results) == 1
        assert results[0].boxes.cls[0] == 0
        assert results[0].names == {0: "cat", 1: "dog"}
        
        # Verify inferencer was called with the temp config
        mock_inferencer_cls.assert_called_once()
        _, kwargs = mock_inferencer_cls.call_args
        assert kwargs["model"] == str(custom_detector.config_path)
        assert kwargs["weights"] == str(custom_checkpoint)
