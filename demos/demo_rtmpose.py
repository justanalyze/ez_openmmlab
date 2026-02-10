from typing import List

from loguru import logger

from ez_openmmlab import RTMPose
from ez_openmmlab.core.inference.results import InferenceResult

# RTMPose is a TOP-DOWN model.
model = RTMPose(model="rtmpose_tiny", log_level="INFO")

image_path = "./demos/demo.jpg"

# predict() handles single image or list of images
# Now always returns a List[InferenceResult]
results: List[InferenceResult] = model.predict(
    image_path=image_path,
    det_model="rtmdet_tiny",
    device="cpu",
    show=True,
    out_dir="./runs/demo_rtmpose",
    bbox_thr=0.4,
    kpt_thr=0.5,
)

# Take the first result since we only passed one image
result = results[0]

if result.keypoints is None:
    logger.warning("No keypoints detected.")
else:
    logger.info(f"Estimated pose for {len(result.keypoints)} people")

    for i in range(len(result.keypoints)):
        logger.info(
            f"Person {i}: Overall Keypoint Scores: {result.keypoints.conf[i]}, Keypoints_coords: {result.keypoints.xy[i]} Keypoints: {len(result.keypoints.xy[i])}"
        )
