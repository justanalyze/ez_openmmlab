from ez_openmmlab import RTMDet

# 1. Select your base architecture
model = RTMDet(model="rtmdet_tiny")

# 2. Path to your simple data definition
dataset_toml = "tests/data/coco_mini/dataset.toml"

# 3. Start training
model.train(
    dataset_config_path=dataset_toml,
    input_size=(320, 320),
    work_dir="./runs/rtmdet_sample_training",
    device="cpu",
    amp=False,
    epochs=5,
    batch_size=2,
    num_workers=2,
)
