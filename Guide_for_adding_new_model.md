# Model Integration Guide

This guide outlines the standard procedure for integrating a new OpenMMLab model into the `ez_openmmlab` framework, ensuring it adheres to the project's **SOLID** architecture and **Context-Aware** philosophy.

---

## Phase 1: Foundation (Schemas & Constants)

Before writing the model logic, you must register its identity and configuration mapping.

1.  **Update `src/ez_openmmlab/schemas/model.py`**:
    *   Add a new entry to the `ModelName` Enum.
    *   Create a dictionary mapping the Enum name to its official OpenMMLab `.py` config path (relative to the library's `configs/` root).
    *   Add the new mapping to the `SUPPORTED_CONFIGS` aggregate.

---

## Phase 2: Parameter Intelligence (Resolvers)

If your model requires derived parameters (e.g., calculating heatmap sigma from input resolution), implement a Resolver.

1.  **Create a Resolver in `src/ez_openmmlab/core/resolvers/`**:
    *   Inherit from `BaseModelParamsResolver`.
    *   Implement `resolve(**kwargs)`: This method should handle defaults and validate parameter compatibility (e.g., ensuring resolution is 32-divisible).
2.  **Register in `ModelParamsResolverFactory`**:
    *   Update `src/ez_openmmlab/core/resolvers/factory.py` to return your new resolver based on the model name.

---

## Phase 3: The Model Engine (User Interface)

This is the primary user-facing class. Place it in `src/ez_openmmlab/models/<library_family>/`.

### 1. Inherit from the Family Base
*   Detection: `EZMMDetector`
*   Pose: `EZMMPose`
*   New families: `EZMMLab`

### 2. Implement the "DNA" Methods
*   `_get_architecture_params`: **Critical.** Return a dict of parameters (like `input_size`) that must be saved to `user_config.toml`. These are "experiment-locked" during resumption.
*   `_get_library_family`: Return the string identifier (e.g., `"mmdet"`, `"mmpose"`).

### 3. Implement the Training Interface (Option 3 Strategy)
To adhere to **Interface Segregation**, you must implement two distinct methods:

*   **`train(...)`**: Strictly for fresh starts.
    *   **Signature**: All parameters should have visible default values.
    *   **Requirement**: Must require `dataset_config_path`.
*   **`resume(...)`**: Strictly for continuing unfinished runs.
    *   **Signature**: All parameters must be `Optional` with `None` as defaults.
    *   **Logic**: Call `super().resume()`. This ensures that if a user doesn't provide an override, the engine strictly uses the values recovered from the source TOML.

---

## Phase 4: Inner Wiring (Injectors & Rebinders)

### 1. Value Patching (Injectors)
Modify `MMDetInjector` or `MMPoseInjector` in `src/ez_openmmlab/core/injectors/` if you need to update specific values in the `Config` object (like `out_channels` or global `codec` settings).

### 2. Structural Synchronization (Rebinders)
If your model config "bakes in" copies of objects (e.g., putting the codec inside the head decoder AND inside the pipeline encoder), **do not patch them manually.** 
*   Add a `StructuralRebinder` in `src/ez_openmmlab/core/injectors/structural.py`.
*   This ensures that all internal references point to your patched global variable, preventing "bin-size mismatches" or resolution errors.

---

## Phase 5: Smart Resumption Logic

The base engine handles most of this, but ensure your model directory contains:
1.  **`user_config.toml`**: The source of truth for all parameters.
2.  **`last_checkpoint`** or **`best_*.pth`**: The weights.

When a user calls `Model(model="runs/exp1/user_config.toml").resume()`, the system automatically:
*   Resolves the `work_dir` to `runs/exp1/`.
*   Locates the best/latest checkpoint.
*   Re-generates the finalized `.py` config with `resume=True`.

---

## Implementation Checklist

| Component | Responsibility | File Location |
| :--- | :--- | :--- |
| **Enum** | Identity & Config Path | `schemas/model.py` |
| **Resolver** | Logic for derived params | `core/resolvers/` |
| **Injector** | Patching value in `.py` Config | `core/injectors/` |
| **Rebinder** | Wiring internal config refs | `core/injectors/structural.py` |
| **Model Class** | User UI (Train vs Resume) | `models/<family>/<name>.py` |

---

## The "Golden Rule" of Resumption
Never rely on method defaults inside `resume()`. By using `None` as defaults and recovering state from `self._source_toml`, we guarantee that a resumed experiment is mathematically identical to the original, even if the library code's default values change in the future.
