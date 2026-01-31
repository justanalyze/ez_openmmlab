from Cython.Compiler.Pythran import is_type
from ez_openmmlab import RTMO
from loguru import logger

# RTMO is a BOTTOM-UP model.
model = RTMO("rtmo_s")

image_path = "./demos/demo.jpg"
image_path_2 = "tests/data/coco_mini/images/000000000389.jpg"

# Run batch inference with RTMO
results = model.predict(
    image_path=[image_path, image_path_2],
    device="cpu",
    show=True,
    out_dir="./runs/demo_rtmo_batch",
    bbox_thr=0.5,
    kpt_thr=0.5,
)

logger.info(f"Bottom-up pose estimation complete for {len(results)} images.")  # type: ignore

for img_idx, res in enumerate(results):
    logger.info(f"Image {img_idx} found {len(res.keypoints)} people")
    for i in range(len(res.keypoints)):
        logger.info(f"Person {i} keypoints: {res.keypoints.xy[i]}")
