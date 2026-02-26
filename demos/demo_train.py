from ez_openmmlab import RTMDet

# 1. Select your base architecture
model = RTMDet(model="rtmdet_tiny")

# 2. Path to your simple data definition
dataset_toml = "tests/data/coco_mini/dataset.toml"

# 3. Start training
model.train(
    dataset_config_path=dataset_toml,
    input_size=(320, 320),
    work_dir="./runs/rtmdet_sample_training_v3",
    device="cpu",
    amp=False,
    epochs=100,
    batch_size=2,
    num_workers=2,
    enable_tensorboard=False,
    augments={"scale_factor": (0.567, 1.523), "random_flip_prob": 0.432},
    stage2_num_epochs=50,
    dry_run=True,
)
