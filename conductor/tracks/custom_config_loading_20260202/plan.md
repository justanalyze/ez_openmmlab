# Implementation Plan: Support Custom Model Loading via config.toml

## Phase 1: Core Base Enhancements

- [x] Task: Update `EZMMLab` constructor signature [04bcbd9]
    - [ ] Modify `__init__` in `src/ez_openmmlab/core/base.py` to accept `model: Union[str, Path, ModelName]`.
    - [ ] Update type hints and docstrings.
- [ ] Task: Implement Custom Config Loading Logic
    - [ ] Create private method `_resolve_model_config(self, model_input: Union[str, Path, ModelName])`.
    - [ ] Implement logic:
        - [ ] If input is `ModelName` (or string equivalent): Use existing default behavior (fetch base weights/config).
        - [ ] If input is path to `.toml`:
            - [ ] Validate existence of `checkpoint_path`. Raise `ValueError` if missing.
            - [ ] Parse `config.toml` to get `model.name` and other settings.
            - [ ] Resolve base config path using `ModelName(model_name).config_path`.
            - [ ] Create a temporary `.py` config file that inherits from the base config and applies overrides from `.toml`.
            - [ ] Return the path to this temporary `.py` file as the config to be used.
- [ ] Task: Cleanup Mechanism
    - [ ] Ensure temporary `.py` files are deleted after initialization (or use `tempfile` module contexts).
- [ ] Task: Conductor - User Manual Verification 'Core Base Enhancements' (Protocol in workflow.md)

## Phase 2: Engine Updates

- [ ] Task: Update `EZMMDetector`
    - [ ] Ensure `EZMMDetector` inherits the new constructor logic.
    - [ ] Verify `_init_inferencer` uses the resolved config path.
- [ ] Task: Update `EZMMPose`
    - [ ] Ensure `EZMMPose` (and subclasses `RTMPose`, `RTMO`) inherits the new constructor logic.
    - [ ] Verify `_init_inferencer` uses the resolved config path.
- [ ] Task: Conductor - User Manual Verification 'Engine Updates' (Protocol in workflow.md)

## Phase 3: Artifact Management

- [ ] Task: Update Config Saving
    - [ ] Modify `save_user_config` (or call site in `base.py`) to include `model.base_config_path` in the saved `user_config.toml`.
    - [ ] Ensure this path resolves to the absolute path of the generated `.py` config file created in the workdir during training.
- [ ] Task: Conductor - User Manual Verification 'Artifact Management' (Protocol in workflow.md)

## Phase 4: Verification

- [ ] Task: Unit Tests
    - [ ] Test `EZMMLab` constructor with valid `ModelName`.
    - [ ] Test `EZMMLab` constructor with `config.toml` + `checkpoint`.
    - [ ] Test `EZMMLab` constructor with `config.toml` missing `checkpoint` (expect failure).
- [ ] Task: Integration Test
    - [ ] Simulate a full flow: Train a model -> Save `config.toml` -> Load new model instance using that `config.toml` + checkpoint -> Run inference.
- [ ] Task: Conductor - User Manual Verification 'Verification' (Protocol in workflow.md)
