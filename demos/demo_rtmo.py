from typing import List

from loguru import logger

from ez_openmmlab import RTMO
from ez_openmmlab.core.inference import InferenceResult

# RTMO is a BOTTOM-UP model.
model = RTMO("rtmo_s")

image_path = "./demos/demo.jpg"
image_path_2 = "tests/data/coco_mini/images/000000000389.jpg"

# Run batch inference with RTMO
# Now always returns a List[InferenceResult]
results: List[InferenceResult] = model.predict(
    image_path=[image_path, image_path_2],
    device="cpu",
    show=True,
    out_dir="./runs/demo_rtmo_batch",
    bbox_thr=0.5,
    kpt_thr=0.5,
)

logger.info(f"Bottom-up pose estimation complete for {len(results)} images.")

for img_idx, res in enumerate(results):
    if res.keypoints is None:
        logger.info(f"Image {img_idx}: No people detected.")
        continue

    logger.info(f"Image {img_idx} found {len(res.keypoints)} people")
    for i in range(len(res.keypoints)):
        logger.info(f"Person {i} keypoints: {res.keypoints.xy[i]}")
