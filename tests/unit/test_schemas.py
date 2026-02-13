import pytest
from pydantic import ValidationError

from ez_openmmlab.schemas.dataset import DatasetConfig
from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils.toml_config import TrainingSection, UserConfig


def test_training_section_defaults():
    """Verify default values for TrainingSection."""
    ts = TrainingSection()
    assert ts.epochs == 100
    assert ts.batch_size == 8
    assert ts.device == "cuda"
    assert ts.amp is True


def test_user_config_validation():
    """Test UserConfig validation with valid and invalid data."""
    valid_data = {
        "model": {"name": "rtmdet_tiny", "num_classes": 80},
        "data": {"root": "data/coco", "dataset_name": "TestUser"},
        "training": {"epochs": 50},
    }
    config = UserConfig(**valid_data)
    assert config.model.name == ModelName.RTM_DET_TINY
    assert config.training.epochs == 50

    invalid_data = {
        "model": {"name": "invalid_model", "num_classes": 80},
        "data": {"root": "data/coco", "dataset_name": "TestInvalid"},
        "training": {"epochs": 50},
    }
    with pytest.raises(ValidationError):
        UserConfig(**invalid_data)


def test_dataset_config_validation():
    """Test DatasetConfig validation."""
    valid_data = {
        "data_root": "datasets/my_data",
        "dataset_name": "TestDS",
        "train": {"ann_file": "train.json", "img_dir": "train/"},
        "val": {"ann_file": "val.json", "img_dir": "val/"},
    }
    ds = DatasetConfig(**valid_data)
    assert str(ds.data_root) == "datasets/my_data"
    assert ds.dataset_name == "TestDS"
    assert ds.train.ann_file == "train.json"