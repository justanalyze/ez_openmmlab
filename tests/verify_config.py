import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
from ez_openmmlab import RTMDet, RTMPose, RTMO
from mmengine.config import Config

def verify_model_config(model_cls, model_name, dataset_toml, description):
    print(f"\n--- Verifying {description} ({model_name}) ---")
    
    # Mock resources to avoid real downloads or path checks during init
    with patch("ez_openmmlab.core.engines.engine_base.ensure_model_checkpoint", return_value=Path("dummy.pth")):
        with patch("ez_openmmlab.core.engines.engine_base.get_config_file", return_value=Path("dummy.py")):
            model = model_cls(model_name)
            
            work_dir = f"runs/verify_{model_name}"
            # Use dry_run to generate the config file
            model.train(
                dataset_config_path=dataset_toml,
                dry_run=True,
                work_dir=work_dir,
                epochs=100,
                stage2_num_epochs=30
            )
            
            # Manually resolve the config path (model_VerifyDataset.py)
            config_path = Path(work_dir) / f"{model_name}_VerifyDataset.py"
            print(f"Generated config path: {config_path}")
            
            # Load and verify the generated config
            cfg = Config.fromfile(str(config_path))
            
            # Basic common verifications
            assert cfg.train_dataloader is not None
            assert cfg.val_dataloader is not None

            # Verify Stage 2 Switch timing if hook is present
            if hasattr(cfg, "custom_hooks"):
                for hook in cfg.custom_hooks:
                    if "PipelineSwitchHook" in hook.type:
                        # 100 - 30 = 70
                        assert hook.switch_epoch == 70
                        print(f"Verified Stage 2 switch_epoch: {hook.switch_epoch}")
            
            # Model-specific verifications can be added here
            if model_cls == RTMDet:
                # Check for RTMDet specific fields if necessary
                pass
            
            print(f"Verification successful for {model_name}!")
            return config_path

def main():
    # Setup a dummy dataset.toml for verification
    dummy_toml = Path("tests/dummy_verify_dataset.toml")
    dummy_toml.write_text("""
data_root = "tests/data/coco_mini"
dataset_name = "VerifyDataset"
classes = ["person", "dog"]

[train]
ann_file = "annotations/instances_train2017.json"
img_dir = "train2017"

[val]
ann_file = "annotations/instances_val2017.json"
img_dir = "val2017"
""")

    # Mock the internal calls that require real data/registry
    # We mock DynamicDatasetRegistry.register_dataset to avoid OpenMMLab registry issues in dry run
    with patch("ez_openmmlab.core.engines.engine_base.DynamicDatasetRegistry.register_dataset", return_value="VerifyDataset"):
        # We mock _load_base_config because we are using dummy.py
        with patch("ez_openmmlab.core.engines.engine_base.EZMMLab._load_base_config", return_value=Config(dict(
            model=dict(bbox_head=dict(num_classes=80), head=dict(out_channels=17)),
            train_dataloader=dict(dataset=dict(pipeline=[])),
            val_dataloader=dict(dataset=dict(pipeline=[])),
            optim_wrapper=dict(optimizer=dict()),
            custom_hooks=[dict(type="PipelineSwitchHook", switch_epoch=80)]
        ))):
            
            try:
                verify_model_config(RTMDet, "rtmdet_tiny", dummy_toml, "Object Detection")
                verify_model_config(RTMPose, "rtmpose_s", dummy_toml, "Top-down Pose")
                verify_model_config(RTMO, "rtmo_s", dummy_toml, "Bottom-up Pose")
                
                print("\nALL CONFIG VERIFICATIONS PASSED!")
            finally:
                if dummy_toml.exists():
                    dummy_toml.unlink()

if __name__ == "__main__":
    main()
