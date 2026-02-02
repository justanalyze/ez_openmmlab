from ez_openmmlab import RTMPose
from ez_openmmlab.schemas.model import ModelName

# 1. Initialize RTMPose with the tiny variant
model = RTMPose(model_name=ModelName.RTM_POSE_TINY)

# 2. Path to our micro dataset definition
dataset_toml = "datasets/micro_cub/dataset.toml"

# 3. Start training
# We'll use a small number of epochs and batch size for this micro dataset
model.train(
    dataset_config_path=dataset_toml,
    work_dir="./runs/train_micro_cub",
    device="cpu",  # Change to "cuda" if you have a GPU
    amp=False,
    epochs=10,
    batch_size=2,
    num_workers=2,
    learning_rate=0.004,
)
