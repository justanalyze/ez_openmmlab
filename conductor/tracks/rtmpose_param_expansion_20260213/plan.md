# Implementation Plan: RTMPose Training Parameter Expansion

## Phase 1: SOLID Refactoring and Advanced Derivation
- [x] Task: Create `src/ez_openmmlab/core/derivers/` architectural foundation
    - [x] Implement `BaseParameterDeriver` abstract class.
    - [x] Implement `RTMPoseParameterDeriver` with resolution adjustment and sigma scaling logic.
    - [x] Implement `DefaultParameterDeriver` for standard models.
    - [x] Implement `DeriverFactory` to resolve the correct deriver by model name.
- [x] Task: Refactor `ConfigManager.build_user_config` to delegate derivation to the factory.
- [x] Task: TDD - Write unit tests for `RTMPoseParameterDeriver`
    - [x] Verify auto-adjustment to nearest multiple of 32.
    - [x] Verify warning is issued for non-32-divisible inputs.
    - [x] Verify linear scaling of sigma.
- [x] Task: Conductor - User Manual Verification 'Phase 1: SOLID Refactoring and Advanced Derivation' (Protocol in workflow.md)

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