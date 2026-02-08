# Implementation Plan: Architectural Reorganization - Move Engines to Core

## Phase 1: Structural Refactoring

- [x] Task: Relocate Engines and Rename Base Class
    - [x] Create directory `src/ez_openmmlab/core/engines/`
    - [x] Move `src/ez_openmmlab/engines/mmdet.py` to `src/ez_openmmlab/core/engines/mmdet.py`
    - [x] Move `src/ez_openmmlab/engines/mmpose.py` to `src/ez_openmmlab/core/engines/mmpose.py`
    - [x] Move `src/ez_openmmlab/core/base.py` to `src/ez_openmmlab/core/engines/engine_base.py`
- [x] Task: Update Internal Engine Imports
    - [x] Update `mmdet.py` to import `EZMMLab` from `.engine_base`
    - [x] Update `mmpose.py` to import `EZMMLab` from `.engine_base`
- [ ] Task: Conductor - User Manual Verification 'Structural Refactoring' (Protocol in workflow.md)

## Phase 2: Dependency Integration and API Preservation

- [x] Task: Update Global Imports [45a3cd7]
    - [x] Update all models in `src/ez_openmmlab/models/` to import from the new `core.engines` path
    - [x] Update `src/ez_openmmlab/core/config_manager.py` (if it references engines)
    - [x] Update any utility scripts or CLI logic referencing the old `engines` path
- [x] Task: Preserve Public API [45a3cd7]
    - [x] Update `src/ez_openmmlab/__init__.py` to export the specialized engines from their new core locations
- [x] Task: Final Verification and Cleanup [45a3cd7]
    - [x] Delete empty `src/ez_openmmlab/engines/` directory
    - [x] Run full test suite: `uv run pytest`
    - [x] Verify demos and CLI tools
- [ ] Task: Conductor - User Manual Verification 'Dependency Integration and API Preservation' (Protocol in workflow.md)
