from ez_openmmlab import RTMDet

# Initialize the Instance Segmentation variant
model = RTMDet("rtmdet-ins_s")

image_path = "./demos/demo.jpg"
image_path_2 = "tests/data/coco_mini/images/000000000389.jpg"

# Run batch prediction
# Passing a list of images triggers the batch processing pipeline
results = model.predict(
    image_path=[image_path, image_path_2],
    device="cpu",
    show=False,
    confidence=0.5,
    out_dir="./runs/demo_segmentation_batch",
)

for result in results:
    for mask in result.masks:
        print(mask)
