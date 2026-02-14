import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ez_openmmlab.core.config_manager import ConfigManager
from ez_openmmlab.core.derivers.rtmpose import RTMPoseParameterDeriver

@patch("ez_openmmlab.core.derivers.rtmpose.logger")
def test_rtmpose_deriver_resolution_adjustment(mock_logger):
    """Verify auto-adjustment to nearest multiple of 32."""
    deriver = RTMPoseParameterDeriver()
    
    # 1. Test adjustment
    derived = deriver.derive(input_size=(170, 240), simcc_sigma=None, feature_map_size=None)
    assert derived["input_size"] == (160, 256) # 170 -> 160, 240 -> 256
    
    # 2. Test rounding up
    derived = deriver.derive(input_size=(180, 250), simcc_sigma=None, feature_map_size=None)
    assert derived["input_size"] == (192, 256) # 180 -> 192, 250 -> 256
    
    # Verify warning was called for first adjustment
    first_warning = mock_logger.warning.call_args_list[0][0][0]
    assert "(170, 240) -> (160, 256)" in first_warning

    # Verify second warning
    second_warning = mock_logger.warning.call_args_list[1][0][0]
    assert "(180, 250) -> (192, 256)" in second_warning

    # 2. Test already divisible
    mock_logger.reset_mock()
    derived = deriver.derive(input_size=(192, 256), simcc_sigma=None, feature_map_size=None)
    assert derived["input_size"] == (192, 256)
    mock_logger.warning.assert_not_called()

def test_rtmpose_deriver_sigma_scaling():
    """Verify linear scaling of simcc_sigma."""
    deriver = RTMPoseParameterDeriver()
    
    # Double the resolution
    derived = deriver.derive(input_size=(384, 512), simcc_sigma=None, feature_map_size=None)
    assert derived["simcc_sigma"] == (9.8, 11.32) # (4.9*2, 5.66*2)

def test_rtmpose_deriver_feature_map_derivation():
    """Verify 1/32 stride derivation."""
    deriver = RTMPoseParameterDeriver()
    derived = deriver.derive(input_size=(192, 256), simcc_sigma=None, feature_map_size=None)
    assert derived["feature_map_size"] == (6, 8)

def test_build_user_config_delegation():
    """Verify that ConfigManager correctly uses the deriver factory."""
    manager = ConfigManager()
    
    with patch("ez_openmmlab.core.config_manager.DatasetConfig.from_toml") as mock_ds:
        mock_ds.return_value = MagicMock(
            dataset_name="TestDS",
            classes=["cat"],
            data_root="/data",
            train=MagicMock(ann_file="t.json", img_dir="t/"),
            val=MagicMock(ann_file="v.json", img_dir="v/"),
            test=None,
            metainfo=None
        )
        
        # This should trigger RTMPoseParameterDeriver logic
        cfg = manager.build_user_config(
            model="rtmpose_s",
            dataset_config_path="dummy.toml",
            input_size=(170, 240) 
        )
        
        assert cfg.model.input_size == (160, 256)
        assert cfg.model.feature_map_size == (5, 8)
