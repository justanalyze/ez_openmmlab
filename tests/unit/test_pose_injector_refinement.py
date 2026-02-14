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
                input_size=None, # Initial value to confirm setting by injector
                in_featuremap_size=None # Initial value to confirm setting by injector
            )
        ),
        codec=dict(
            type="SimCCLabel",
            input_size=(192, 256), # Initial value before patching
            sigma=(4.9, 5.66)     # Initial value before patching
        ),
        train_pipeline=[
            Config(dict(type="LoadImage")),
            Config(dict(type="TopdownAffine", input_size=None))
        ],
        optim_wrapper=dict(
            optimizer=dict(type="AdamW", weight_decay=0.05)
        ),
        val_evaluator=dict(type="CocoMetric")
    ))

    injector = MMPoseInjector()
    injector.apply(cfg, mock_pose_user_config)

    # 1. Head (num_keypoints is still patched here)
    assert cfg.model.head.out_channels == 17
    assert cfg.model.head.input_size == (288, 384) 
    assert cfg.model.head.in_featuremap_size == (9, 12) 

    # 2. Codec
    assert cfg.codec.input_size == (288, 384)
    assert cfg.codec.sigma == (5.0, 5.0)

    # 3. Pipeline (TopdownAffine input_size is derived from codec)
    assert cfg.train_pipeline[1].input_size == (288, 384)

    # 4. Optimizer (now handled by OptimizerInjector, not MMPoseInjector)
    # assert cfg.optim_wrapper.optimizer.weight_decay == mock_pose_user_config.training.weight_decay

    # 5. Evaluator (now handled by EvaluatorInjector, not MMPoseInjector)
    # assert cfg.val_evaluator.type == "PCKAccuracy"
    assert cfg.val_evaluator.type == "CocoMetric" # Retains initial value as MMPoseInjector no longer patches it