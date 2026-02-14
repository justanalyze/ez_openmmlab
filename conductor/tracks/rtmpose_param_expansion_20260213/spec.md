# Specification: RTMPose Training Parameter Expansion

## Overview
This track extends the `RTMPose` training API to allow fine-grained control over model resolution, SimCC codec parameters, and optimization settings. It introduces a "Resolution-Aware" configuration workflow where `simcc_sigma` and `feature_map_size` are intelligently derived from the `input_size`. 

**Architectural Note:** To adhere to SOLID principles, parameter derivation is decoupled from the `ConfigManager` into dedicated `ParameterDeriver` classes.

## Functional Requirements

### 1. Schema Updates (`UserConfig`)
- Add `input_size` (tuple[int, int]) to the `ModelSection`, defaulting to `(192, 256)`.
- Add `simcc_sigma` (tuple[float, float]) to the `ModelSection`, defaulting to `(4.9, 5.66)`.
- Add `feature_map_size` (Optional[tuple[int, int]]) to the `ModelSection`, defaulting to `None`.
- Add `weight_decay` (float) to the `TrainingSection`, defaulting to `0.05`.
- Add `evaluator_metric` (Union[str, List[str]]) to the `TrainingSection`, defaulting to `"CocoMetric"`.

### 2. RTMPose API Enhancements
- Update `RTMPose.train()` to accept these new parameters as optional arguments.
- Parameters passed to `train()` will override those in the `dataset.toml` or the schema defaults.

### 3. Smart Parameter Logic (via `RTMPoseParameterDeriver`)
- **Resolution Adjustment:** If `input_size` is not perfectly divisible by 32, the system will:
    - Issue a warning to the user.
    - Automatically adjust the `input_size` to the nearest multiple of 32.
- **Sigma Scaling:** If `input_size` is customized but `simcc_sigma` is not, scale the sigma values linearly based on the ratio between the new resolution and the reference `(192, 256)`.
- **Feature Map Derivation:** If `feature_map_size` is `None`, automatically calculate it as `input_size // 32`.
- **Validation:** Ensure `feature_map_size` is mathematically compatible with `simcc_sigma`.

### 4. Configuration Injection
- Update `MMPoseInjector` to correctly patch:
    - `codec.input_size` and `codec.sigma`.
    - `model.head.input_size` and `model.head.in_featuremap_size`.
    - `train_pipeline` and `val_pipeline` affine transforms.
    - `optim_wrapper.optimizer.weight_decay`.
    - `val_evaluator` and `test_evaluator` types.

## Acceptance Criteria
- [ ] `RTMPose.train` automatically adjusts non-32-divisible resolutions and warns the user.
- [ ] `simcc_sigma` is automatically scaled when resolution changes, unless explicitly overridden.
- [ ] `feature_map_size` defaults to the correct 1/32 ratio.
- [ ] All derivation logic is encapsulated in model-specific `Deriver` classes.
- [ ] New parameters are correctly persisted in the `user_config.toml` artifact.