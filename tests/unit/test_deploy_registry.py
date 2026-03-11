import pytest
from ez_openmmlab.core.deploy.registry import DeployConfigRegistry

def test_resolve_mmdet_onnx():
    """Test that mmdet onnx config is correctly resolved."""
    registry = DeployConfigRegistry()
    config_path = registry.get_deploy_cfg(family="mmdet", format="onnx")
    assert "mmdet" in config_path
    assert "onnxruntime" in config_path
    assert config_path.endswith(".py")

def test_resolve_mmdet_tensorrt():
    """Test that mmdet tensorrt config is correctly resolved."""
    registry = DeployConfigRegistry()
    config_path = registry.get_deploy_cfg(family="mmdet", format="tensorrt")
    assert "mmdet" in config_path
    assert "tensorrt" in config_path
    assert config_path.endswith(".py")

def test_resolve_mmpose_onnx():
    """Test that mmpose onnx config is correctly resolved."""
    registry = DeployConfigRegistry()
    config_path = registry.get_deploy_cfg(family="mmpose", format="onnx")
    assert "mmpose" in config_path
    assert "onnxruntime" in config_path
    assert config_path.endswith(".py")

def test_resolve_invalid_format():
    """Test that an invalid format raises a ValueError."""
    registry = DeployConfigRegistry()
    with pytest.raises(ValueError, match="Unsupported format"):
        registry.get_deploy_cfg(family="mmdet", format="invalid")

def test_resolve_invalid_family():
    """Test that an invalid family raises a ValueError."""
    registry = DeployConfigRegistry()
    with pytest.raises(ValueError, match="Unsupported model family"):
        registry.get_deploy_cfg(family="invalid", format="onnx")
