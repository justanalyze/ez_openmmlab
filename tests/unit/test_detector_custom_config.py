from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from ez_openmmlab import RTMDet


@patch("ez_openmmlab.core.engines.mmdet.DetInferencer")
@patch("ez_openmmlab.core.resolvers.resource_resolver.ensure_model_checkpoint")
@patch("ez_openmmlab.core.config_manager.get_config_file")
def test_rtmdet_predict_auto_loads_toml_params(
    mock_get_config, mock_ensure, mock_inferencer_cls, tmp_path
):
    """Verify that RTMDet automatically uses arch params from config.toml."""
    # 1. Setup a fake config.toml with custom resolution
    config_toml = tmp_path / "custom_config.toml"
    config_toml.write_text("""
[model]
name = "rtmdet_tiny"
num_classes = 5
input_size = [320, 320]

[data]
root = "."

[training]
epochs = 1
batch_size = 1
""")
    
    mock_ensure.return_value = tmp_path / "dummy.pth"
    mock_get_config.return_value = tmp_path / "dummy.py"
    # Create a dummy python config that the injector will patch
    (tmp_path / "dummy.py").write_text("model = dict(bbox_head=dict(type='RTMDetHead'))\ntest_pipeline = [dict(type='Resize', scale=(640, 640))]")

    mock_inferencer_instance = MagicMock()
    mock_inferencer_cls.return_value = mock_inferencer_instance

    with patch("ez_openmmlab.core.engines.mmdet.Config.fromfile") as mock_fromfile:
        real_cfg = MagicMock()
        real_cfg.model.bbox_head = MagicMock()
        real_cfg.test_pipeline = [dict(type="Resize", scale=(640, 640))]
        mock_fromfile.return_value = real_cfg
        
        with patch("cv2.imread", return_value=np.zeros((480, 640, 3), dtype=np.uint8)):
            # Load model from the config.toml
            model = RTMDet(model=config_toml, checkpoint_path=tmp_path / "m.pth")
            
            # Call predict
            model.predict("test.jpg")

    # 2. Verify that the config passed to DetInferencer was patched
    _, kwargs = mock_inferencer_cls.call_args
    passed_cfg = kwargs.get("model")
    
    # num_classes should be 5
    assert passed_cfg.model.bbox_head.num_classes == 5
    # input_size should be (320, 320) in the pipeline
    assert passed_cfg.test_pipeline[0]["scale"] == (320, 320)
