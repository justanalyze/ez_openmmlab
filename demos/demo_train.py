from pathlib import Path

from ez_openmmlab import RTMDet

# 1. Select your base architecture
model = RTMDet(model="rtmdet_tiny")

# 2. Path to your simple data definition
dataset_toml = "tests/data/coco_mini/dataset.toml"
dataset_toml_micro_cub = "datasets/micro_cub/dataset.toml"

if not Path(dataset_toml).exists():
    print(f"Skipping training demo: {dataset_toml} not found.")
else:
    # 3. Start training
    model.train(
        dataset_config_path=dataset_toml,
        work_dir="./runs/demo_rtmdet_cub_train",
        device="cpu",
        amp=False,
        epochs=5,
        batch_size=2,
        num_workers=2,
    )
