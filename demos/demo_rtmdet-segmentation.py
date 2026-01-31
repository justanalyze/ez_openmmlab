from ez_openmmlab import RTMDet
from loguru import logger

# Initialize the Instance Segmentation variant
model = RTMDet("rtmdet-ins_s")

image_path = "./demos/demo.jpg"
image_path_2 = "tests/data/coco_mini/images/000000000389.jpg"

# Run batch prediction
# Passing a list of images triggers the batch processing pipeline
results = model.predict(
    image_path=[
        image_path,
        image_path_2,
    ],  # This line will be updated by the replace operation
    device="cpu",
    show=False,
    confidence=0.5,
    out_dir="./runs/demo_segmentation_batch",
)

if isinstance(results, list):
    logger.info(f"Segmented instances across {len(results)} images")

    # Accessing the first image's results
    first_image_results = results[0]
    for i in range(len(first_image_results.boxes)):
        logger.info(
            f"Image 0, Instance [{i}] Label: {int(first_image_results.boxes.cls[i])}, Score: {first_image_results.boxes.conf[i]:.2f}"
        )
else:
    logger.info(f"Segmented {len(results.boxes)} instances")
