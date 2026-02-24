import pytest
from unittest.mock import MagicMock, patch
from ez_openmmlab.models.mmdet import RTMDet
from ez_openmmlab.models.mmpose import RTMPose
from ez_openmmlab.core.schema.models import ModelName

class ConcreteEZDetector(RTMDet):
    def _init_inferencer(self, device, **kwargs): pass
    def _get_architecture_params(self, **kwargs) -> dict: return {}

class ConcreteEZPose(RTMPose):
    def _init_inferencer(self, device, **kwargs): pass
    def _get_architecture_params(self, **kwargs) -> dict: return {}
    def _instantiate_inferencer(self, cfg, device, **kwargs): pass

def test_train_unsupported_augmentation_raises_error():
    """Verifies that providing an invalid augmentation key raises ValueError."""
    # Note: We need to mock enough of the init to avoid real file checks if possible,
    # or just use a valid model name and mock the registry.
    with patch("ez_openmmlab.core.engines.engine_base.ensure_model_checkpoint", return_value="dummy.pth"):
        with patch("ez_openmmlab.core.engines.engine_base.get_config_file", return_value="dummy.py"):
            detector = ConcreteEZDetector(model=ModelName.RTM_DET_TINY)
            
            with pytest.raises(ValueError, match="Unsupported augmentation key"):
                detector.train(
                    dataset_config_path="dummy.toml",
                    augments={"invalid_key": 1.0}
                )
