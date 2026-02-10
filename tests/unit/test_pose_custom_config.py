from pathlib import Path
from unittest.mock import MagicMock, patch

from ez_openmmlab import RTMO, RTMPose
from ez_openmmlab.schemas.model import ModelName


@patch("ez_openmmlab.core.config_manager.ConfigManager.load_metadata_from_toml")
@patch("ez_openmmlab.models.mmpose.rtmpose.MMPoseInferencer")
@patch("ez_openmmlab.core.config_manager.toml_config.load_user_config")
@patch("ez_openmmlab.core.engines.engine_base.get_config_file")
@patch("ez_openmmlab.core.engines.engine_base.ensure_model_checkpoint")
def test_rtmpose_init_with_custom_config(mock_ensure, mock_get_config, mock_load_config, mock_inferencer_cls, mock_load_meta, tmp_path):
    """Test that RTMPose correctly initializes using a custom config.toml."""
    config_file = tmp_path / "config.toml"
    config_file.touch()
    checkpoint_file = tmp_path / "custom.pth"
    checkpoint_file.touch()

    # Setup mocks
    mock_load_meta.return_value = {
        "num_classes": 1,
        "num_keypoints": 15,
        "metainfo": {"classes": ["bird"]},
        "model_name": "rtmpose_tiny"
    }
    mock_user_config = MagicMock()
    mock_user_config.model.name = ModelName.RTM_POSE_TINY
    mock_user_config.model.num_classes = 1
    mock_user_config.model.num_keypoints = 15
    mock_user_config.data.metainfo = {"classes": ["bird"]}
    mock_load_config.return_value = mock_user_config

    mock_get_config.return_value = Path.cwd() / "libs/mmpose/configs/rtmpose/tiny.py"
    mock_ensure.return_value = checkpoint_file

    # Initialize RTMPose
    # Note: EZMMLab._auto_load_metadata might trigger if checkpoint exists
    # but we are mocking the toml load anyway.
    pose = RTMPose(model=config_file, checkpoint_path=checkpoint_file)

    # Trigger _init_inferencer
    with patch("ez_openmmlab.models.mmpose.rtmpose.Config.fromfile") as mock_cfg_fromfile:
        mock_cfg = MagicMock()
        mock_cfg_fromfile.return_value = mock_cfg
        pose._init_inferencer(device="cpu")
        mock_cfg_fromfile.assert_called_once_with(str(pose.config_path))

    # Verify MMPoseInferencer was called with the Config object (patched)
    mock_inferencer_cls.assert_called_once()
    args, kwargs = mock_inferencer_cls.call_args
    assert kwargs["pose2d"] == mock_cfg
    assert str(kwargs["pose2d_weights"]) == str(checkpoint_file)
    # Check if head was patched
    assert mock_cfg.model.head.out_channels == 15

@patch("ez_openmmlab.core.config_manager.ConfigManager.load_metadata_from_toml")
@patch("ez_openmmlab.models.mmpose.rtmo.MMPoseInferencer")
@patch("ez_openmmlab.core.config_manager.toml_config.load_user_config")
@patch("ez_openmmlab.core.engines.engine_base.get_config_file")
@patch("ez_openmmlab.core.engines.engine_base.ensure_model_checkpoint")
def test_rtmo_init_with_custom_config(mock_ensure, mock_get_config, mock_load_config, mock_inferencer_cls, mock_load_meta, tmp_path):
    """Test that RTMO correctly initializes using a custom config.toml."""
    config_file = tmp_path / "config.toml"
    config_file.touch()
    checkpoint_file = tmp_path / "custom.pth"
    checkpoint_file.touch()

    # Setup mocks
    mock_load_meta.return_value = {
        "num_classes": 1,
        "num_keypoints": 15,
        "metainfo": {"classes": ["bird"]},
        "model_name": "rtmo_s"
    }
    mock_user_config = MagicMock()
    mock_user_config.model.name = ModelName.RTMO_S
    mock_user_config.model.num_classes = 1
    mock_user_config.model.num_keypoints = 15
    mock_user_config.data.metainfo = {"classes": ["bird"]}
    mock_load_config.return_value = mock_user_config

    mock_get_config.return_value = Path.cwd() / "libs/mmpose/configs/rtmo/s.py"
    mock_ensure.return_value = checkpoint_file

    # Initialize RTMO
    pose = RTMO(model=config_file, checkpoint_path=checkpoint_file)

    # Trigger _init_inferencer
    with patch("ez_openmmlab.models.mmpose.rtmo.Config.fromfile") as mock_cfg_fromfile:
        mock_cfg = MagicMock()
        mock_cfg_fromfile.return_value = mock_cfg
        pose._init_inferencer(device="cpu")
        mock_cfg_fromfile.assert_called_once_with(str(pose.config_path))

    # Verify MMPoseInferencer was called
    mock_inferencer_cls.assert_called_once()
    args, kwargs = mock_inferencer_cls.call_args
    assert kwargs["pose2d"] == mock_cfg
    # Check if head was patched
    assert mock_cfg.model.head.num_keypoints == 15
