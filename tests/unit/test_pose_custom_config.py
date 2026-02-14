from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ez_openmmlab import RTMO, RTMPose


@patch("ez_openmmlab.models.mmpose.rtmpose.MMPoseInferencer")
@patch("ez_openmmlab.core.engines.engine_base.ensure_model_checkpoint")
@patch("ez_openmmlab.core.engines.engine_base.get_config_file")
def test_rtmpose_predict_patches_arch_params(
    mock_get_config, mock_ensure, mock_inferencer_cls, tmp_path
):
    """Verify that RTMPose.predict patches architecture params into the config."""
    mock_ensure.return_value = tmp_path / "dummy.pth"
    mock_get_config.return_value = tmp_path / "dummy.py"
    (tmp_path / "dummy.py").write_text("model = dict(head=dict(type='RTMCCHead'))\ncodec=dict(type='SimCCLabel')")
    
    mock_inferencer_instance = MagicMock()
    mock_inferencer_instance.return_value = iter([])
    mock_inferencer_cls.return_value = mock_inferencer_instance

    with patch("ez_openmmlab.core.engines.mmpose.Config.fromfile") as mock_fromfile:
        # Create a mock config that looks like a real one
        real_cfg = MagicMock()
        real_cfg.model.head = MagicMock()
        real_cfg.codec = MagicMock()
        mock_fromfile.return_value = real_cfg
        
        with patch("cv2.imread", return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
            # We must patch _run_inference because RTMPose is abstract in this test context
            # if we don't use the real implementation or mock it correctly.
            # However, RTMPose actually implements it via EZMMPose.
            model = RTMPose("rtmpose_s", num_keypoints=17)
            
            # Call predict with custom architecture params
            model.predict("test.jpg", input_size=(288, 384), simcc_sigma=(5.0, 5.0))

    # Verify that the config passed to MMPoseInferencer was patched
    # MMPoseInferencer(pose2d=cfg, ...) -> pose2d is a keyword arg or first positional
    _, kwargs = mock_inferencer_cls.call_args
    passed_cfg = kwargs.get("pose2d")
    
    assert passed_cfg.model.head.input_size == (288, 384)
    assert passed_cfg.codec.sigma == (5.0, 5.0)


@patch("ez_openmmlab.models.mmpose.rtmo.MMPoseInferencer")
@patch("ez_openmmlab.core.engines.engine_base.ensure_model_checkpoint")
@patch("ez_openmmlab.core.engines.engine_base.get_config_file")
def test_rtmo_predict_patches_input_size(
    mock_get_config, mock_ensure, mock_inferencer_cls, tmp_path
):
    """Verify that RTMO.predict patches input_size into the config."""
    mock_ensure.return_value = tmp_path / "dummy.pth"
    mock_get_config.return_value = tmp_path / "dummy.py"
    (tmp_path / "dummy.py").write_text("model = dict(head=dict(type='RTMOHead'))")
    
    mock_inferencer_instance = MagicMock()
    mock_inferencer_instance.return_value = iter([])
    mock_inferencer_cls.return_value = mock_inferencer_instance

    with patch("ez_openmmlab.core.engines.mmpose.Config.fromfile") as mock_fromfile:
        real_cfg = MagicMock()
        real_cfg.model.head = MagicMock()
        mock_fromfile.return_value = real_cfg
        
        with patch("cv2.imread", return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
            model = RTMO("rtmo_s", num_keypoints=17)
            model.predict("test.jpg", input_size=(640, 640))

    _, kwargs = mock_inferencer_cls.call_args
    passed_cfg = kwargs.get("pose2d")
    
    assert passed_cfg.model.head.input_size == (640, 640)


@patch("ez_openmmlab.models.mmpose.rtmpose.MMPoseInferencer")
@patch("ez_openmmlab.core.engines.engine_base.ensure_model_checkpoint")
@patch("ez_openmmlab.core.engines.engine_base.get_config_file")
def test_rtmpose_predict_auto_loads_toml_params(
    mock_get_config, mock_ensure, mock_inferencer_cls, tmp_path
):
    """Verify that RTMPose automatically uses arch params from config.toml without explicit predict() args."""
    # 1. Setup a fake config.toml with custom resolution
    config_toml = tmp_path / "custom_config.toml"
    config_toml.write_text("""
[model]
name = "rtmpose_s"
num_classes = 1
num_keypoints = 17
input_size = [160, 160]
simcc_sigma = [3.0, 3.0]

[data]
root = "."

[training]
epochs = 1
batch_size = 1
""")
    
    mock_ensure.return_value = tmp_path / "dummy.pth"
    mock_get_config.return_value = tmp_path / "dummy.py"
    (tmp_path / "dummy.py").write_text("model = dict(head=dict(type='RTMCCHead'))\ncodec=dict(type='SimCCLabel')")

    mock_inferencer_instance = MagicMock()
    mock_inferencer_instance.return_value = iter([])
    mock_inferencer_cls.return_value = mock_inferencer_instance

    with patch("ez_openmmlab.core.engines.mmpose.Config.fromfile") as mock_fromfile:
        real_cfg = MagicMock()
        real_cfg.model.head = MagicMock()
        real_cfg.codec = MagicMock()
        mock_fromfile.return_value = real_cfg
        
        with patch("cv2.imread", return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
            # Load model from the config.toml
            model = RTMPose(model=config_toml, checkpoint_path=tmp_path / "m.pth")
            
            # Call predict WITHOUT architecture params
            model.predict("test.jpg")

    # 2. Verify that the inferencer received the patched config with TOML values
    _, kwargs = mock_inferencer_cls.call_args
    passed_cfg = kwargs.get("pose2d")
    
    # Values should match the TOML, not the defaults
    assert passed_cfg.model.head.input_size == [160, 160]
    assert passed_cfg.codec.sigma == [3.0, 3.0]
