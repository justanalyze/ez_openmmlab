import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from ez_openmmlab.core.config_manager import ConfigManager
from ez_openmmlab.core.resolvers.rtmpose import RTMPoseParamsResolver

@patch("ez_openmmlab.core.resolvers.rtmpose.logger")
def test_rtmpose_resolver_resolution_adjustment(mock_logger):
    """Verify auto-adjustment to nearest multiple of 32."""
    resolver = RTMPoseParamsResolver()
    
    # 1. Test adjustment
    resolved = resolver.resolve(input_size=(170, 240))
    assert resolved["input_size"] == (160, 256) # 170 -> 160, 240 -> 256
    
    # 2. Test rounding up
    resolved = resolver.resolve(input_size=(180, 250))
    assert resolved["input_size"] == (192, 256) # 180 -> 192, 250 -> 256
    
    # Verify warning was called for first adjustment
    first_warning = mock_logger.warning.call_args_list[0][0][0]
    assert "(170, 240) -> (160, 256)" in first_warning

    # Verify second warning
    second_warning = mock_logger.warning.call_args_list[1][0][0]
    assert "(180, 250) -> (192, 256)" in second_warning

    # 3. Test already divisible
    mock_logger.reset_mock()
    resolved = resolver.resolve(input_size=(192, 256))
    assert resolved["input_size"] == (192, 256)
    mock_logger.warning.assert_not_called()

def test_rtmpose_resolver_sigma_scaling():
    """Verify linear scaling of simcc_sigma."""
    resolver = RTMPoseParamsResolver()
    
    # Double the resolution
    resolved = resolver.resolve(input_size=(384, 512))
    assert resolved["simcc_sigma"] == (9.8, 11.32) # (4.9*2, 5.66*2)

def test_rtmpose_resolver_feature_map_derivation():
    """Verify 1/32 stride derivation."""
    resolver = RTMPoseParamsResolver()
    resolved = resolver.resolve(input_size=(192, 256))
    assert resolved["feature_map_size"] == (6, 8)

def test_build_user_config_delegation():
    """Verify that ConfigManager correctly uses the resolver factory."""
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
        
        # This should trigger RTMPoseParamsResolver logic
        cfg = manager.build_user_config(
            model="rtmpose_s",
            dataset_config_path="dummy.toml",
            architecture_params={"input_size": (170, 240)},
            weight_decay=0.01,
            evaluator_metric="PCKAccuracy"
        )
        
        assert cfg.model.input_size == (160, 256)
        assert cfg.model.feature_map_size == (5, 8)
        assert cfg.training.weight_decay == 0.01
        assert cfg.training.evaluator_metric == "PCKAccuracy"