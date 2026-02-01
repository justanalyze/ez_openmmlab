from typing import List
from ez_openmmlab import RTMDet
from ez_openmmlab.core.results import InferenceResult
from loguru import logger

# 1. Simple Initialization
model = RTMDet("rtmdet_s")

# 2. Path to our sample image
image_path = "./demos/demo.jpg"

# 3. Batch Inference Example
# ez_openmmlab supports passing a list of images for faster processing
images = [image_path, image_path]
batch_results: List[InferenceResult] = model.predict(
    image_path=images,
    device="cpu",
    show=False,
    confidence=0.5,
    out_dir="./runs/demo_detection_batch",
)

logger.info(f"Batch inference complete. Processed {len(batch_results)} images.")

# 4. Single Image Inference with Structured Result Access
results: List[InferenceResult] = model.predict(
    image_path=image_path,
    device="cpu",
    show=False,
    confidence=0.5,
    out_dir="./runs/demo_detection_single",
)

result = results[0]

if result.boxes is None:
    logger.info("No objects detected.")
else:
    logger.info(f"Single detection found {len(result.boxes)} objects")
    for i in range(len(result.boxes)):
        logger.info(
            f"[{i}] Label ID: {int(result.boxes.cls[i])}, Score: {result.boxes.conf[i]:.2f}, BBox: {result.boxes.xyxy[i]}"
        )
