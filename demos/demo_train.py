from torch.cpu.amp import amp
from ez_openmmlab import RTMDet

# 1. Choose your base architecture (rtmdet, rtmpose, etc.)
model = RTMDet("rtmdet_tiny")

# 2. Point to your simple dataset definition (TOML-first approach)
dataset_toml = "tests/data/coco_mini/dataset.toml"

# 3. Start training on your custom data
model.train(
    dataset_config_path=dataset_toml,
    epochs=100,
    batch_size=8,
    device="cpu",  # Change to "cuda" if you have a GPU
    work_dir="runs/rtmdet_training_demo",
    # Powerful data augmentation simplified to a few keys
    augments={
        "scale_factor": (0.5, 2.0),
        "random_flip_prob": 0.5,
    },
    amp=False,
    # dry_run=True,  # Set to False to actually start training. This is very useful if you want to verify the correctness of your training config first.
)
