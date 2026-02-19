import pytest
from mmengine.config import Config

from ez_openmmlab.core.surgery.injectors.dataloader import DataloaderInjector
from ez_openmmlab.core.surgery.injectors.runtime import RuntimeInjector
from ez_openmmlab.utils.toml_config import (
    DataSection,
    ModelSection,
    TrainingSection,
    UserConfig,
)


@pytest.fixture
def mock_user_config():
    return UserConfig(
        model=ModelSection(name="rtmdet_tiny", num_classes=2),
        data=DataSection(
            root="data/coco",
            train_ann="annotations/train.json",
            train_img="train2017",
            val_ann="annotations/val.json",
            val_img="val2017",
            classes=["cat", "dog"],
        ),
        training=TrainingSection(
            epochs=10,
            batch_size=4,
            num_workers=2,
            learning_rate=0.01,
            amp=True,
            enable_tensorboard=True,
        ),
    )


def test_dataloader_injector_applies_paths_and_params(mock_user_config):
    """Test that DataloaderInjector correctly sets paths and parameters."""
    mock_user_config.data.registered_class_name = "MyDynamicDataset"
    cfg = Config(dict(
        train_dataloader=dict(dataset=dict()),
        val_dataloader=dict(dataset=dict())
    ))

    injector = DataloaderInjector()
    injector.apply(cfg, mock_user_config)

    assert cfg.data_root == "data/coco"
    assert cfg.train_dataloader.num_workers == 2
    assert cfg.train_dataloader.dataset.type == "MyDynamicDataset"
    assert cfg.train_dataloader.dataset.ann_file == "data/coco/annotations/train.json"


def test_dataloader_injector_no_classes(mock_user_config):
    """Test injector when classes list is None."""
    mock_user_config.data.classes = None
    cfg = Config(dict(
        train_dataloader=dict(dataset=dict())
    ))

    injector = DataloaderInjector()
    injector.apply(cfg, mock_user_config)

    assert "metainfo" not in cfg.train_dataloader.dataset
    assert "metainfo" not in cfg


def test_runtime_injector_applies_settings(mock_user_config):
    """Test that RuntimeInjector correctly sets optimizer, visualizer, and runtime params."""
    cfg = Config(dict(
        train_cfg=dict(),
        optim_wrapper=dict(optimizer=dict()),
        visualizer=dict(vis_backends=[])
    ))

    injector = RuntimeInjector()
    injector.apply(cfg, mock_user_config)

    assert cfg.work_dir == "./runs/train"
    assert cfg.train_cfg.max_epochs == 10
    assert cfg.optim_wrapper.type == "AmpOptimWrapper"

    # Check TensorBoard
    backends = [b["type"] for b in cfg.visualizer.vis_backends]
    assert "TensorboardVisBackend" in backends


def test_runtime_injector_disabled_features(mock_user_config):
    """Test injector when AMP and TensorBoard are disabled."""
    mock_user_config.training.amp = False
    mock_user_config.training.enable_tensorboard = False
    cfg = Config(dict(
        train_cfg=dict(),
        optim_wrapper=dict(optimizer=dict())
    ))

    injector = RuntimeInjector()
    injector.apply(cfg, mock_user_config)

    assert cfg.optim_wrapper.type == "OptimWrapper"
    assert "visualizer" not in cfg


def test_runtime_injector_missing_optim_wrapper(mock_user_config):
    """Test that injector doesn't crash if optim_wrapper is missing."""
    cfg = Config(dict(train_cfg=dict()))

    injector = RuntimeInjector()
    injector.apply(cfg, mock_user_config) # Should not raise

    assert cfg.work_dir == "./runs/train"


def test_evaluator_injector_handles_dicts(mock_user_config):
    """Test that EvaluatorInjector correctly processes string and dict metrics."""
    from ez_openmmlab.core.surgery.injectors.evaluator import EvaluatorInjector

    # 1. Mixed metrics: string and dict
    mock_user_config.training.evaluator_metric = [
        "CocoMetric",
        {"type": "PCKAccuracy", "thr": 0.05}
    ]
    
    cfg = Config(dict(
        val_evaluator=dict(type="StandardMetric", ann_file="old.json"),
        test_evaluator=dict(type="StandardMetric")
    ))

    injector = EvaluatorInjector()
    injector.apply(cfg, mock_user_config)

    # Both evaluators should now be lists of 2 metrics
    assert len(cfg.val_evaluator) == 2
    assert cfg.val_evaluator[0]["type"] == "CocoMetric"
    assert cfg.val_evaluator[0]["ann_file"] == "data/coco/annotations/val.json"
    
    assert cfg.val_evaluator[1]["type"] == "PCKAccuracy"
    assert cfg.val_evaluator[1]["thr"] == 0.05
    assert cfg.val_evaluator[1]["ann_file"] == "data/coco/annotations/val.json"

    # 2. Single dict metric
    mock_user_config.training.evaluator_metric = {"type": "PCKAccuracy", "thr": 0.1}
    injector.apply(cfg, mock_user_config)
    
    # Should be a single dict (not a list) for compatibility
    assert isinstance(cfg.val_evaluator, dict)
    assert cfg.val_evaluator["type"] == "PCKAccuracy"
    assert cfg.val_evaluator["thr"] == 0.1
