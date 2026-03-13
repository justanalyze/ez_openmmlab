from pathlib import Path
from unittest.mock import MagicMock, patch

from ez_openmmlab.core.engines.engine_base import EZMMLab
from ez_openmmlab.core.inference.results import InferenceResult


class MockFormatter:
    def map_results(self, results, inputs, names):
        return [InferenceResult(path=inputs[0], names=names)]


class ConcreteMockEngine(EZMMLab):
    def __init__(self, model="rtmdet_tiny"):
        # We must patch the resolver's dependencies now
        with patch(
            "ez_openmmlab.core.resolvers.resource_resolver.ensure_model_checkpoint",
            return_value=Path("mock.pth"),
        ):
            with patch(
                "ez_openmmlab.core.config_manager.get_config_file",
                return_value=Path("mock.py"),
            ):
                super().__init__(model=model)
        self._formatter = MockFormatter()
        self._inferencer = MagicMock()
        self.init_called = False
        self.run_called = False

    def _init_inferencer(self, device: str, **kwargs):
        self.init_called = True
        self.device = device

    def _get_library_family(self) -> str:
        return "mmdet"

    def _get_architecture_params(self, **kwargs) -> dict:
        return {}

    def _run_inference(self, inputs: list, out_dir: str, show: bool, **kwargs):
        self.run_called = True
        self.kwargs = kwargs
        return {"raw": "data"}


def test_predict_template_workflow():
    """Verify that EZMMLab.predict calls the hooks in the correct order."""
    engine = ConcreteMockEngine()

    with patch("ez_openmmlab.core.engines.engine_base.normalize_inputs", return_value=["img.jpg"]):
        results = engine.predict("img.jpg", device="cpu", confidence=0.5)

        assert engine.init_called
        assert engine.run_called
        assert engine.device == "cpu"
        assert engine.kwargs["confidence"] == 0.5
        assert len(results) == 1
        assert results[0].path == "img.jpg"


def test_get_class_names_logic():
    """Verify that _get_class_names respects metainfo priority."""
    engine = ConcreteMockEngine()

    # 1. Test metainfo priority
    engine.metainfo = {"classes": ["cat", "dog"]}
    assert engine._get_class_names() == {0: "cat", 1: "dog"}

    # 2. Test inferencer fallback
    engine.metainfo = None
    mock_model = MagicMock()
    mock_model.dataset_meta = {"classes": ["person"]}
    engine._inferencer.model = mock_model
    assert engine._get_class_names() == {0: "person"}


def test_export_orchestration(tmp_path):
    """Test that export() correctly orchestrates the Docker process."""
    engine = ConcreteMockEngine()
    engine.checkpoint_path = tmp_path / "best.pth"
    engine.checkpoint_path.touch()
    
    # We must provide a valid minimal config structure for the injectors to patch
    engine.config_path = tmp_path / "mock.py"
    engine.config_path.write_text("model = dict(bbox_head=dict(num_classes=80))")
    
    # We need to mock the internals of export
    with patch("ez_openmmlab.core.deploy.docker_manager.DockerExportManager") as mock_manager_cls, \
         patch("ez_openmmlab.core.deploy.registry.DeployConfigRegistry") as mock_registry_cls, \
         patch("ez_openmmlab.core.config_manager.get_config_file") as mock_get_config:
        
        mock_get_config.return_value = Path("mock.py")
        mock_registry = mock_registry_cls.return_value
        mock_registry.get_deploy_cfg.return_value = "configs/deploy.py"
        
        mock_manager = mock_manager_cls.return_value
        mock_manager.build_command.return_value = "docker run ..."
        
        # Call export
        engine.export(format="onnx", image="test.jpg", output_dir=str(tmp_path / "export"))
        
        # Verify interactions
        mock_registry.get_deploy_cfg.assert_called_with("mmdet", "onnx", model_name="rtmdet_tiny")
        mock_manager.build_command.assert_called()
        mock_manager.run_export.assert_called_with("docker run ...")
