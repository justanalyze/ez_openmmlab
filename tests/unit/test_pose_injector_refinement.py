import pytest
from mmengine.config import Config
from ez_openmmlab.core.injectors.mmpose import MMPoseInjector
from ez_openmmlab.utils.toml_config import (
    DataSection,
    ModelSection,
    TrainingSection,
    UserConfig,
)

@pytest.fixture
def mock_pose_user_config():
    return UserConfig(
        model=ModelSection(
            name="rtmpose_tiny", 
            num_classes=1, 
            num_keypoints=17,
            input_size=(288, 384),
            simcc_sigma=(5.0, 5.0),
            feature_map_size=(9, 12)
        ),
        data=DataSection(root="."),
        training=TrainingSection(
            weight_decay=0.01,
            evaluator_metric="PCKAccuracy"
        )
    )

def test_mmpose_injector_comprehensive_patching(mock_pose_user_config):
    """Verify that all RTMPose hyperparameters are correctly injected."""
    cfg = Config(dict(
        model=dict(
            head=dict(
                type="RTMCCHead",
                out_channels=17,
                input_size=(192, 256),
                in_featuremap_size=(6, 8)
            )
        ),
        codec=dict(
            type="SimCCLabel",
            input_size=(192, 256),
            sigma=(4.9, 5.66)
        ),
        train_pipeline=[
            dict(type="LoadImage"),
            dict(type="TopdownAffine", input_size=(192, 256))
        ],
        optim_wrapper=dict(
            optimizer=dict(type="AdamW", weight_decay=0.05)
        ),
        val_evaluator=dict(type="CocoMetric")
    ))

    injector = MMPoseInjector()
    injector.apply(cfg, mock_pose_user_config)

    # 1. Head
    assert cfg.model.head.input_size == (288, 384)
    assert cfg.model.head.in_featuremap_size == (9, 12)

    # 2. Codec
    assert cfg.codec.input_size == (288, 384)
    assert cfg.codec.sigma == (5.0, 5.0)

    # 3. Pipeline
    assert cfg.train_pipeline[1].input_size == (288, 384)

    # 4. Optimizer
    assert cfg.optim_wrapper.optimizer.weight_decay == 0.01

    # 5. Evaluator
    assert cfg.val_evaluator.type == "PCKAccuracy"
