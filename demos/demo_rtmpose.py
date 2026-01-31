from ez_openmmlab import RTMPose
from loguru import logger
from ez_openmmlab.schemas.model import ModelName

# RTMPose is a TOP-DOWN model.
model = RTMPose(model_name=ModelName.RTM_POSE_S, log_level="DEBUG")

image_path = "./demos/demo.jpg"

# predict() handles single image or list of images
result = model.predict(
    image_path=image_path,
    det_model="rtmdet_tiny",
    device="cpu",
    show=True,
    out_dir="./runs/demo_rtmpose",
    bbox_thr=0.4,
    kpt_thr=0.5,
)

logger.info(f"Estimated pose for {len(result.keypoints)} people")

for i in range(len(result.keypoints)):
    logger.info(
        f"Person {i}: Overall Keypoint Scores: {result.keypoints.conf[i]}, Keypoints_coords: {result.keypoints.xy[i]} Keypoints: {len(result.keypoints.xy[i])}"
    )
