# Implementation Plan: Consolidate and Refine Configuration Management

## Phase 1: Module Consolidation and Refinement [checkpoint: 1d2ba9b]
Focus on creating the new `config_manager.py` and consolidating logic.

- [x] Task: Create `src/ez_openmmlab/core/config_manager.py` and Implement Base Logic
    - [x] Create `tests/unit/test_config_manager.py` with failing tests for consolidated logic
    - [x] Implement `BaseConfigLoader` logic in `config_manager.py`
    - [x] Implement `ConfigManager` (renamed from `UserConfigBuilder`) in `config_manager.py`
    - [x] Implement `get_config_file` utility in `config_manager.py`
    - [x] Move `cleanup_temp_config` and `prepare_config_file` to `ConfigManager`
- [x] Task: Apply Optimizations and Renaming
    - [x] Update `load_metadata_from_checkpoint` to search only `checkpoint_path.parent` for `user_config.toml`
    - [x] Update `prepare_config_file` to use module-qualified `toml_config` calls
    - [x] Rename `ConfigManager.build()` to `ConfigManager.build_user_config()`
    - [x] Verify unit tests for `ConfigManager` pass
- [x] Task: Conductor - User Manual Verification 'Phase 1: Module Consolidation and Refinement' (Protocol in workflow.md)

## Phase 2: Integration and Cleanup

- [ ] Task: Update Internal References
    - [ ] Update all imports in `src/ez_openmmlab/core/base.py` to point to `config_manager`
    - [ ] Update engines and models referencing `get_config_file` to use the new path
    - [ ] Update all tests to use the consolidated `config_manager`
- [ ] Task: Final Cleanup and Regression Testing
    - [ ] Delete `src/ez_openmmlab/core/config_loader.py`
    - [ ] Delete `src/ez_openmmlab/core/config_builder.py`
    - [ ] Run full project test suite (`uv run pytest`) to ensure no regressions
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Integration and Cleanup' (Protocol in workflow.md)
