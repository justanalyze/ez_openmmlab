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

    def __getitem__(self, index): """Allows for slicing and boolean indexing."""
        return self.__class__(self.data[index], self.orig_shape)

    def __repr__(self):
        return f"{self.__class__.__name__}(len={len(self)}, shape={self.data.shape})"

class Boxes(BaseData):
    """Vectorized bounding boxes.

    Data format: [N, 6] -> [x1, y1, x2, y2, score, label]
    """

    @property
    def xyxy(self) -> np.ndarray:
        return self.data[..., :4]

    @property
    def conf(self) -> np.ndarray:
        return self.data[..., 4]

    @property
    def cls(self) -> np.ndarray:
        return self.data[..., 5]


class Keypoints(BaseData):
    """Vectorized keypoints.

    Data format: [N, K, 3] -> [x, y, score]
    """

    @property
    def xy(self) -> np.ndarray:
        return self.data[..., :2]

    @property
    def conf(self) -> np.ndarray:
        return self.data[..., 2]


class Masks(BaseData):
    """Vectorized segmentation masks.

    Data format: [N, H, W]
    """

    @property
    def xy(self) -> List[np.ndarray]:
        """Returns polygon segments for each mask."""
        segments = []
        # Handle both [N, H, W] and [H, W] (after indexing)
        masks = self.data if self.data.ndim == 3 else [self.data]

        for mask in masks:
            mask = mask.astype(np.uint8) * 255
            contours, _ = cv2.findContours(
                mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
            )
            if contours:
                segments.append(contours[0].reshape(-1, 2))
            else:
                segments.append(np.zeros((0, 2)))
        return segments

class InferenceResult:
    """Standardized container for all inference results."""

    def __init__(
        self,
        path: str,
        names: dict,
        orig_img: Optional[np.ndarray] = None,
        orig_shape: Optional[tuple] = None,
        boxes: Optional[Boxes] = None,
        keypoints: Optional[Keypoints] = None,
        masks: Optional[Masks] = None,
        speed: Optional[dict] = None,
    ):
        self._orig_img = orig_img
        self.path = path
        self.names = names
        self.speed = speed or {}

        # 1. Resolve original shape
        if orig_shape:
            self.orig_shape = orig_shape
        elif orig_img is not None:
            self.orig_shape = orig_img.shape[:2]
        else:
            # Fallback to loading image header/info if absolutely needed,
            # but for now we assume shape was provided if image wasn't.
            self.orig_shape = (0, 0)

        # 2. Ensure data containers are never None (for better DX and linting)
        self.boxes: Boxes = (
            boxes if boxes is not None else Boxes(np.zeros((0, 6)), self.orig_shape)
        )
        self.keypoints: Keypoints = (
            keypoints
            if keypoints is not None
            else Keypoints(np.zeros((0, 0, 3)), self.orig_shape)
        )
        self.masks: Masks = (
            masks
            if masks is not None
            else Masks(np.zeros((0, 0, 0), dtype=bool), self.orig_shape)
        )

    @property
    def orig_img(self) -> np.ndarray:
        """Lazily loads the original image if not already present."""
        if self._orig_img is None:
            self._orig_img = cv2.imread(self.path)
            if self._orig_img is None:
                raise FileNotFoundError(f"Could not read image at: {self.path}")
        return self._orig_img

    def plot(self, line_width: int = 2, font_size: float = 0.5) -> np.ndarray:
        """Plots boxes, masks, and keypoints on the original image."""
        img = self.orig_img.copy()

        # 1. Draw Masks
        for seg in self.masks.xy:
            if seg.size > 0:
                cv2.polylines(
                    img, [seg.astype(np.int32)], True, (0, 255, 0), thickness=line_width
                )

        # 2. Draw Boxes
        for i in range(len(self.boxes)):
            box = self.boxes.xyxy[i].astype(np.int32)
            conf = self.boxes.conf[i]
            cls_id = int(self.boxes.cls[i])
            name = self.names.get(cls_id, str(cls_id))

            label = f"{name} {conf:.2f}"
            cv2.rectangle(
                img, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), line_width
            )
            cv2.putText(
                img,
                label,
                (box[0], box[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_size,
                (255, 0, 0),
                line_width,
            )

        # 3. Draw Keypoints
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
