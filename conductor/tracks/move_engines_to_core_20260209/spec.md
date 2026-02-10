# Specification: Architectural Reorganization - Move Engines to Core

## Overview
Reorganize the project structure to improve cohesion and encapsulation by moving the `engines` module into the `core` package. This aligns the abstract orchestration logic (`EZMMLab`) with its concrete implementations and streamlines the top-level package namespace.

## Goals
- Relocate `src/ez_openmmlab/engines/` to `src/ez_openmmlab/core/engines/`.
- Move and rename `src/ez_openmmlab/core/base.py` to `src/ez_openmmlab/core/engines/engine_base.py`.
- Update all internal imports to reflect the new structure.
- Maintain the user-facing API by ensuring `RTMDet`, `RTMPose`, etc., remain exportable from the root `ez_openmmlab` package.

## Functional Requirements
1.  **Refactor Directory Structure:**
    - Create `src/ez_openmmlab/core/engines/`.
    - Move contents of `src/ez_openmmlab/engines/` into the new directory.
    - Move `src/ez_openmmlab/core/base.py` to `src/ez_openmmlab/core/engines/engine_base.py`.
    - Delete the empty `src/ez_openmmlab/engines/` directory.
2.  **Update Internal References:**
    - Update all engines (`mmdet.py`, `mmpose.py`) to import `EZMMLab` from `.engine_base`.
    - Update all models (`models/**`) to import their respective engine bases from the new `core.engines` path.
    - Update CLI and utility scripts.
3.  **Preserve Public API:**
    - Ensure `src/ez_openmmlab/__init__.py` correctly points to the new locations so that root-level imports continue to work.

## Non-Functional Requirements
- **Consistency:** All core execution logic is now centralized within `core/`.
- **Transparency:** The structural change should be invisible to the end user.

## Acceptance Criteria
- [ ] No files remain in the top-level `src/ez_openmmlab/engines/` directory.
- [ ] `EZMMLab` is located at `src/ez_openmmlab/core/engines/engine_base.py`.
- [ ] `uv run pytest` passes all unit and integration tests.
- [ ] Demos and CLI tools run without "ModuleNotFoundError".

## Out of Scope
- Changing the logic of the engines or the base class.
- Refactoring the `models/` or `utils/` directories.
