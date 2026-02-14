from pathlib import Path

import pytest
import tomli
from pydantic import ValidationError

from ez_openmmlab.schemas.model import ModelName
from ez_openmmlab.utils.toml_config import (
    DataSection,
    ModelSection,
    TrainingSection,
    UserConfig,
    load_user_config,
    save_user_config,
)


@pytest.fixture
def sample_config():
    """Provides a basic UserConfig for testing."""
    return UserConfig(
        model=ModelSection(
            name=ModelName.RTM_DET_TINY,
            num_classes=2,
        ),
        data=DataSection(
            root="tests/dummy_data/crayfish",
            classes=["crayfish", "scallop"],
        ),
        training=TrainingSection(
            epochs=2,
            batch_size=1,
            device="cpu",
            work_dir="output_test",
        ),
    )


def test_user_config_initialization(sample_config: UserConfig):
    """Verifies that UserConfig initializes correctly from components."""
    assert sample_config.model.name == "rtmdet_tiny"
    assert sample_config.data.root == "tests/dummy_data/crayfish"
    assert sample_config.training.epochs == 2


def test_save_and_load_config(tmp_path: Path, sample_config: UserConfig):
    """Verifies that saving and loading results in an equivalent object."""
    config_path = tmp_path / "config.toml"
    save_user_config(sample_config, config_path)

    assert config_path.exists()

    loaded_config = load_user_config(config_path)
    assert loaded_config.model.name == sample_config.model.name
    assert loaded_config.data.root == sample_config.data.root
    assert loaded_config.training.epochs == sample_config.training.epochs
    assert loaded_config.data.classes == ["crayfish", "scallop"]


def test_generated_toml_content(tmp_path: Path, sample_config: UserConfig):
    """Verifies that the generated TOML has the correct structure and values."""
    config_path = tmp_path / "config.toml"
    save_user_config(sample_config, config_path)

    with open(config_path, "rb") as f:
        data = tomli.load(f)

    # Check sections
    assert "model" in data
    assert "data" in data
    assert "training" in data

    # Check specific values
    assert data["model"]["name"] == "rtmdet_tiny"
    assert data["model"]["num_classes"] == 2
    assert "load_from" not in data["model"]  # None should be excluded

    assert data["data"]["root"] == "tests/dummy_data/crayfish"
    assert data["data"]["classes"] == ["crayfish", "scallop"]

    assert data["training"]["epochs"] == 2
    assert data["training"]["device"] == "cpu"
    assert data["training"]["work_dir"] == "output_test"


def test_missing_required_field_raises_error(tmp_path: Path):
    """Ensures that loading a TOML with missing required fields raises a Pydantic error."""
    bad_toml_content = """
    [model]
    name = "rtmdet_tiny"
    num_classes = 80

    [data]
    # root is missing

    [training]
    epochs = 10
    """
    config_path = tmp_path / "bad_config.toml"
    config_path.write_text(bad_toml_content)

    with pytest.raises(ValidationError):
        load_user_config(config_path)


def test_invalid_enum_raises_error(tmp_path: Path):
    """Ensures that invalid model names raise a Pydantic error."""
    bad_toml_content = """
    [model]
    name = "invalid_model_name"
    num_classes = 80

    [data]
    root = "some/path"

    [training]
    epochs = 10
    """
    config_path = tmp_path / "bad_config.toml"
    config_path.write_text(bad_toml_content)

    with pytest.raises(ValidationError):
        load_user_config(config_path)
