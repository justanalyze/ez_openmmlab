from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from ez_openmmlab.cli import app

runner = CliRunner()


def test_cli_help():
    """Test that the CLI provides help output."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Usage:" in result.stdout


def test_train_command_requires_model_name():
    """Test that the train command fails if model_name is missing."""
    result = runner.invoke(app, ["train"])
    assert result.exit_code != 0
    assert "Missing argument" in result.output
    assert "model_name" in result.output.lower()


def test_train_command_requires_dataset_config():
    """Test that the train command fails if dataset_config_path is missing."""
    result = runner.invoke(app, ["train", "rtmdet_tiny"])
    assert result.exit_code != 0
    assert "Missing argument 'DATASET_CONFIG'" in result.output


def test_predict_command_calls_factory_and_predict(tmp_path):
    """Test that the predict command uses ModelFactory and calls predict."""
    checkpoint = tmp_path / "best.pth"
    checkpoint.touch()
    image = tmp_path / "demo.jpg"
    image.touch()

    with patch("ez_openmmlab.cli.ModelFactory.get_model") as mock_get_model:
        mock_model_instance = MagicMock()
        mock_get_model.return_value = mock_model_instance
        mock_model_instance.predict.return_value = [MagicMock()]

        result = runner.invoke(
            app,
            [
                "predict",
                "rtmdet_tiny",
                str(image),
                "--weights",
                str(checkpoint),
                "--out",
                "output",
                "--no-show",
            ],
        )

        assert result.exit_code == 0
        # Verify factory was called with correct params
        mock_get_model.assert_called_once()
        _, kwargs = mock_get_model.call_args
        assert kwargs["checkpoint_path"] == Path(checkpoint)

        # Verify predict was called
        mock_model_instance.predict.assert_called_once()
        _, p_kwargs = mock_model_instance.predict.call_args
        assert p_kwargs["device"] == "cuda"  # Default


def test_train_command_calls_factory_and_train(tmp_path):
    """Test that the train command uses ModelFactory and calls train."""
    dataset_toml = tmp_path / "dataset.toml"
    dataset_toml.touch()

    with patch("ez_openmmlab.cli.ModelFactory.get_model") as mock_get_model:
        mock_model_instance = MagicMock()
        mock_get_model.return_value = mock_model_instance

        result = runner.invoke(
            app,
            [
                "train",
                "rtmdet_tiny",
                str(dataset_toml),
                "--epochs",
                "5",
                "--batch-size",
                "4",
                "--num_workers",
                "0",
            ],
        )

        assert result.exit_code == 0
        mock_get_model.assert_called_once_with("rtmdet_tiny")
        mock_model_instance.train.assert_called_once()
        _, t_kwargs = mock_model_instance.train.call_args
        assert t_kwargs["epochs"] == 5
        assert t_kwargs["batch_size"] == 4
        assert t_kwargs["num_workers"] == 0
