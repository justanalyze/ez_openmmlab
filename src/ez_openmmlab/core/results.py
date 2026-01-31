import numpy as np
import cv2
from typing import Optional, List, Union

class BaseData:
    """Base class for vectorized inference data."""
    def __init__(self, data: np.ndarray, orig_shape: tuple):
        self.data = data
        self.orig_shape = orig_shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Allows for slicing and boolean indexing."""
        return self.__class__(self.data[index], self.orig_shape)

class Boxes(BaseData):
    """Vectorized bounding boxes.
    
    Data format: [N, 6] -> [x1, y1, x2, y2, score, label]
    """
    @property
    def xyxy(self) -> np.ndarray:
        return self.data[:, :4]

    @property
    def conf(self) -> np.ndarray:
        return self.data[:, 4]

    @property
    def cls(self) -> np.ndarray:
        return self.data[:, 5]

class Keypoints(BaseData):
    """Vectorized keypoints.
    
    Data format: [N, K, 3] -> [x, y, score]
    """
    @property
    def xy(self) -> np.ndarray:
        return self.data[:, :, :2]

    @property
    def conf(self) -> np.ndarray:
        return self.data[:, :, 2]

class Masks(BaseData):
    """Vectorized segmentation masks.
    
    Data format: [N, H, W]
    """
    @property
    def xy(self) -> List[np.ndarray]:
        """Returns polygon segments for each mask."""
        segments = []
        for mask in self.data:
            mask = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                segments.append(contours[0].reshape(-1, 2))
            else:
                segments.append(np.zeros((0, 2)))
        return segments

class InferenceResult:
    """Standardized container for all inference results."""
    def __init__(
        self,
        orig_img: np.ndarray,
        path: str,
        names: dict,
        boxes: Optional[Boxes] = None,
        keypoints: Optional[Keypoints] = None,
        masks: Optional[Masks] = None,
        speed: Optional[dict] = None,
    ):
        self.orig_img = orig_img
        self.orig_shape = orig_img.shape[:2]
        self.path = path
        self.names = names
        self.boxes = boxes
        self.keypoints = keypoints
        self.masks = masks
        self.speed = speed or {}

    def plot(self, line_width: int = 2, font_size: float = 0.5) -> np.ndarray:
        """Plots boxes, masks, and keypoints on the original image."""
        img = self.orig_img.copy()
        
        # 1. Draw Masks
        if self.masks is not None:
            for seg in self.masks.xy:
                if seg.size > 0:
                    cv2.polylines(img, [seg.astype(np.int32)], True, (0, 255, 0), thickness=line_width)

        # 2. Draw Boxes
        if self.boxes is not None:
            for i in range(len(self.boxes)):
                box = self.boxes.xyxy[i].astype(np.int32)
                conf = self.boxes.conf[i]
                cls_id = int(self.boxes.cls[i])
                name = self.names.get(cls_id, str(cls_id))
                
                label = f"{name} {conf:.2f}"
                cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), line_width)
                cv2.putText(img, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, font_size, (255, 0, 0), line_width)

        # 3. Draw Keypoints
        if self.keypoints is not None:
            for i in range(len(self.keypoints)):
                for kpt in self.keypoints.xy[i]:
                    cv2.circle(img, (int(kpt[0]), int(kpt[1])), 3, (0, 0, 255), -1)

        return img

    def show(self, title: str = "Inference Result"):
        """Displays the plotted image in a window."""
        img = self.plot()
        cv2.imshow(title, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
