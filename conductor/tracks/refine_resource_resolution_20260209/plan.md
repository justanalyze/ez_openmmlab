# Implementation Plan: Refine Resource Resolution and Enforce Explicit Configuration

## Phase 1: Update ConfigManager and Regression Tests [checkpoint: 45a3cd7]

- [x] Task: Update `ConfigManager` API [1d2ba9b]
    - [x] Create `tests/unit/test_config_manager_refinement.py` verifying the new `load_metadata_from_toml` method.
    - [x] Implement `load_metadata_from_toml(self, path: Path) -> Dict[str, Any]` in `src/ez_openmmlab/core/config_manager.py`.
    - [x] Remove the old `load_metadata_from_checkpoint` method and its broad directory search logic.
- [x] Task: Update Initialization Unit Tests [45a3cd7]
    - [x] Update `tests/unit/test_engine_base_init.py` to reflect the new explicit resolution behavior.
    - [x] Remove tests that rely on "auto-discovery" of metadata from parent directories.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Update ConfigManager and Regression Tests' (Protocol in workflow.md) [45a3cd7]

## Phase 2: Refactor EZMMLab and Final Cleanup [checkpoint: 45a3cd7]

- [x] Task: Refactor `EZMMLab._resolve_resources` [45a3cd7]
    - [x] Update `engine_base.py`:
        - [x] Rewrite `_resolve_resources` with clean branching logic.
        - [x] Move model state assignment (classes, keypoints, names) into the appropriate branches.
        - [x] Remove the call to `_initialize_model_state`.
    - [x] Remove the empty/obsolete `_initialize_model_state` method.
- [x] Task: Final Verification [45a3cd7]
    - [x] Run full project test suite: `uv run pytest`.
    - [x] Verify that custom model loading works strictly via the provided `.toml`.
- [x] Task: Conductor - User Manual Verification 'Refactor EZMMLab and Final Cleanup' (Protocol in workflow.md) [45a3cd7]