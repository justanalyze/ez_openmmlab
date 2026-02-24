from pathlib import Path
import pytest
import tomli
from pydantic import ValidationError
from ez_openmmlab.core.schema.models import ModelName
from ez_openmmlab.core.schema.config import (
    DataSection,
    ModelSection,
    TrainingSection,
    AugmentationSection,
    UserConfig,
    load_user_config,
    save_user_config,
)

@pytest.fixture
def config_with_augments():
    """Provides a UserConfig with augmentation settings."""
    return UserConfig(
        model=ModelSection(name=ModelName.RTM_DET_TINY, num_classes=2),
        data=DataSection(root="data", classes=["a", "b"]),
        training=TrainingSection(epochs=1),
        augments=AugmentationSection(
            scale_factor=(0.5, 1.5),
            rotate_factor=30.0,
            random_flip_prob=0.5
        )
    )

def test_augmentation_section_validation():
    """Verifies that AugmentationSection validates its fields."""
    # Valid
    aug = AugmentationSection(scale_factor=0.8, rotate_factor=45.0, random_flip_prob=0.2)
    assert aug.scale_factor == 0.8
    assert aug.rotate_factor == 45.0
    assert aug.random_flip_prob == 0.2

    # Invalid random_flip_prob
    with pytest.raises(ValidationError):
        AugmentationSection(random_flip_prob=1.5)
    
    # Invalid rotate_factor
    with pytest.raises(ValidationError):
        AugmentationSection(rotate_factor=-10.0)

def test_user_config_with_augments_serialization(tmp_path: Path, config_with_augments: UserConfig):
    """Verifies that UserConfig with augments serializes to TOML correctly."""
    config_path = tmp_path / "config_aug.toml"
    save_user_config(config_with_augments, config_path)

    with open(config_path, "rb") as f:
        data = tomli.load(f)

    assert "augments" in data
    assert data["augments"]["scale_factor"] == [0.5, 1.5]
    assert data["augments"]["rotate_factor"] == 30.0
    assert data["augments"]["random_flip_prob"] == 0.5

def test_load_config_with_augments(tmp_path: Path):
    """Verifies that UserConfig can be loaded from TOML with an [augments] section."""
    toml_content = """
    [model]
    name = "rtmdet_tiny"
    num_classes = 2

    [data]
    root = "data"
    classes = ["a", "b"]

    [training]
    epochs = 1

    [augments]
    scale_factor = 0.9
    rotate_factor = 15.0
    random_flip_prob = 0.3
    """
    config_path = tmp_path / "load_aug.toml"
    config_path.write_text(toml_content)

    loaded = load_user_config(config_path)
    assert loaded.augments is not None
    assert loaded.augments.scale_factor == 0.9
    assert loaded.augments.rotate_factor == 15.0
    assert loaded.augments.random_flip_prob == 0.3
