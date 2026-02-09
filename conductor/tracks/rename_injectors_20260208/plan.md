# Implementation Plan: Terminology Alignment - Rename Handlers to Injectors

## Phase 1: Refactor Injector Infrastructure

- [x] Task: Rename Handlers Directory and Base Classes
    - [x] Write tests verifying current handler behavior (to ensure parity after rename)
    - [x] Rename `src/ez_openmmlab/core/handlers/` to `src/ez_openmmlab/core/injectors/`
    - [x] Rename `BaseConfigHandler` to `BaseConfigInjector` in `base.py`
    - [x] Rename `DataloaderHandler` and `RuntimeHandler` in `common.py`
    - [x] Rename `MMDetHandler` in `mmdet.py` and `MMPoseHandler` in `mmpose.py`
    - [x] Update `src/ez_openmmlab/core/injectors/__init__.py`:
        - [x] Rename `get_handlers` to `get_injectors`
        - [x] Update all internal variable names from `handler` to `injector`
- [x] Task: Conductor - User Manual Verification 'Refactor Injector Infrastructure' (Protocol in workflow.md)

## Phase 2: Engine Integration and Verification

- [x] Task: Update Engines to Use Injectors
    - [x] Update `src/ez_openmmlab/core/base.py` to import from `ez_openmmlab.core.injectors`
    - [x] Replace `get_handlers` calls with `get_injectors`
    - [x] Update loop variables: `for handler in get_handlers(...)` -> `for injector in get_injectors(...)`
- [x] Task: Final Verification and Cleanup
    - [x] Run full test suite: `uv run pytest`
    - [x] Verify that no references to "Handler" (in the context of config patching) remain in `src/ez_openmmlab/core/`
    - [x] Verify training and inference demos
- [x] Task: Conductor - User Manual Verification 'Engine Integration and Verification' (Protocol in workflow.md)
