# Implementation Plan: RTMPose Training Parameter Expansion

## Phase 1: Schema and Validation
- [x] Task: Update Pydantic schemas in `src/ez_openmmlab/utils/toml_config.py`
    - [x] Add `input_size`, `simcc_sigma`, and `feature_map_size` to `ModelSection`.
    - [x] Add `weight_decay` and `evaluator_metric` to `TrainingSection`.
- [x] Task: Implement smart parameter derivation in `ConfigManager.build_user_config`
    - [x] Add logic to scale `simcc_sigma` linearly if not provided.
    - [x] Add logic to derive `feature_map_size` (input_size // 32) if not provided.
    - [x] Add validation check for sigma/feature map compatibility.
- [x] Task: TDD - Write unit tests for parameter derivation and schema validation
    - [x] Verify default values.
    - [x] Verify linear scaling of sigma for non-standard resolutions.
    - [x] Verify auto-calculation of feature map size.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Schema and Validation' (Protocol in workflow.md)

## Phase 2: Configuration Injection
- [ ] Task: Update `MMPoseInjector` in `src/ez_openmmlab/core/injectors/mmpose.py`
    - [ ] Patch `codec.input_size` and `codec.sigma`.
    - [ ] Patch `model.head.input_size` and `model.head.in_featuremap_size`.
    - [ ] Patch `optim_wrapper.optimizer.weight_decay`.
    - [ ] Patch `val_evaluator` and `test_evaluator` with the provided metric list.
- [ ] Task: TDD - Write unit tests for `MMPoseInjector`
    - [ ] Verify that OpenMMLab Config objects are correctly modified with the new parameters.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Configuration Injection' (Protocol in workflow.md)

## Phase 3: RTMPose API
- [ ] Task: Update `RTMPose.train` in `src/ez_openmmlab/models/mmpose/rtmpose.py`
    - [ ] Update signature to accept new parameters.
    - [ ] Ensure they are passed down to `build_user_config`.
- [ ] Task: Update `EZMMLab.train` and `ConfigManager` to support the extended parameter set
- [ ] Task: TDD - Create an integration test for RTMPose custom training
    - [ ] verify that a training run can start with custom `input_size` and `evaluator_metric`.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: RTMPose API' (Protocol in workflow.md)
