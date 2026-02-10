from pathlib import Path
from typing import List

from loguru import logger

from ez_openmmlab import RTMPose
from ez_openmmlab.core.inference.results import InferenceResult
from ez_openmmlab.schemas.model import ModelName

# 1. Initialize RTMPose with the tiny variant and our new checkpoint

# Use absolute path to ensure it's found regardless of working directory switches

checkpoint = str(Path("runs/train_micro_cub/best_coco_AP_epoch_10.pth").absolute())

# Metadata (num_keypoints, etc.) is now auto-loaded from user_config.toml nearby!

model = RTMPose(
    model_name=ModelName.RTM_POSE_TINY,
    checkpoint_path=checkpoint,
)

# 2. Path to one of our micro images
image_path = "/home/kalebtata/Projects/ez_mmdet/datasets/micro_cub/images/Rufous_Hummingbird_0075_59619.jpg"

# 3. Predict
# Since we trained on a custom dataset, we need to pass det_cat_ids=[0]
# (assuming the detector finds the bird as person/class 0, or we use a bird detector)
# For this demo, let's just see if it runs.
results: List[InferenceResult] = model.predict(
    image_path=image_path,
    device="cpu",
    show=False,
    out_dir="./runs/predict_micro_cub",
    det_cat_ids=[0],
    det_model="rtmdet_tiny",
    det_weights="./runs/demo_rtmdet_cub_train/epoch_5.pth",
)

result = results[0]
if result.keypoints is not None:
    logger.info(f"Estimated pose for {len(result.keypoints)} birds")
    for i in range(len(result.keypoints)):
        logger.info(f"Bird {i} keypoints: {result.keypoints.xy[i]}")
else:
    logger.warning("No birds detected or keypoints estimated.")
