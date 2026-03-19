# Packaging Refactor - Code Review TODO

## High Priority
- [x] **ConfigManager.load_base_config (L129)**: Fix `relative_to()` fragility. Handle cases where `config_path` is not a subpath of `lib_root` (e.g. cross-mount or different roots) by falling back to absolute path.
    ```python
    try:
        rel_config_path = config_path.relative_to(lib_root)
        return Config.fromfile(str(rel_config_path))
    except ValueError:
        return Config.fromfile(str(config_path))
    ```

## Medium Priority
- [ ] **BaseConfigLoader._resolve_package_configs (L21)**: Simplify and make package config resolution more robust using `importlib.resources` or strictly adhering to `.mim/configs` convention.
- [ ] **EZMMLab._load_and_patch_config (L327)**: Validate the assumption that setting `dataset_name = None` during inference is safe for all custom dataset scenarios. Monitor for cases where specialized dataset classes provide metadata required by the inferencer.
- [ ] **publish.sh (L35)**: Downgrade `setuptools` pin from `==80` to `<70` to ensure broader compatibility and stable `pkg_resources` availability for `mmengine`.

## Low Priority (Optional)
- [ ] **DynamicDatasetRegistry (L71)**: Add debug logging when skipping registration for an already registered dataset.
- [ ] **pyproject.toml (L29)**: Pin `chumpy` git dependency to a specific commit hash for better reproducibility.
