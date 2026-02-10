from pathlib import Path
from unittest.mock import MagicMock, patch

from ez_openmmlab.core.engines.engine_base import EZMMLab
from ez_openmmlab.core.results import InferenceResult


class MockFormatter:
    def map_results(self, results, inputs, names):
        return [InferenceResult(path=inputs[0], names=names)]


class ConcreteMockEngine(EZMMLab):
    def __init__(self, model="mock_model"):
        # We need to bypass some of the real resource resolution for testing the template logic
        with patch(
            "ez_openmmlab.core.engines.engine_base.ensure_model_checkpoint",
            return_value=Path("mock.pth"),
        ):
            with patch(
                "ez_openmmlab.core.engines.engine_base.get_config_file",
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