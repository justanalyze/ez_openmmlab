from typing import Any, Callable, List, Optional, Union

import cv2
import numpy as np


class BaseData:
    """Base class for vectorized inference data."""

    def __init__(self, data: np.ndarray, orig_shape: tuple):
        self.data = data
        self.orig_shape = orig_shape

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """Allows for slicing and boolean indexing."""
        new_data = self.data[index]
        return self.__class__(new_data, self.orig_shape)

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
        boxes: Optional[Union[Boxes, Any]] = None,
        keypoints: Optional[Union[Keypoints, Any]] = None,
        masks: Optional[Union[Masks, Any]] = None,
        speed: Optional[dict] = None,
        format_fn: Optional[Callable] = None,
    ):
        self._orig_img = orig_img
        self.path = path
        self.names = names
        self.speed = speed or {}
        self._format_fn = format_fn

        # Internal storage for potentially raw data
        self._boxes_raw = boxes
        self._keypoints_raw = keypoints
        self._masks_raw = masks

        # Cached processed objects
        self._boxes: Optional[Boxes] = None
        self._keypoints: Optional[Keypoints] = None
        self._masks: Optional[Masks] = None

        # Resolve original shape
        if orig_shape:
            self.orig_shape = orig_shape
        elif orig_img is not None:
            self.orig_shape = orig_img.shape[:2]
        else:
            # Fallback to loading image header/info if absolutely needed,
            # but for now we assume shape was provided if image wasn't.
            self.orig_shape = (0, 0)

    @property
    def boxes(self) -> Boxes:
        """Lazily processes and returns bounding boxes."""
        if self._boxes is None:
            if isinstance(self._boxes_raw, Boxes):
                self._boxes = self._boxes_raw
            elif self._format_fn and self._boxes_raw is not None:
                # The format_fn should handle extracting and converting raw data
                self._boxes = self._format_fn(self, "boxes", self._boxes_raw)
            else:
                self._boxes = Boxes(np.zeros((0, 6)), self.orig_shape)
        return self._boxes

    @property
    def keypoints(self) -> Keypoints:
        """Lazily processes and returns keypoints."""
        if self._keypoints is None:
            if isinstance(self._keypoints_raw, Keypoints):
                self._keypoints = self._keypoints_raw
            elif self._format_fn and self._keypoints_raw is not None:
                self._keypoints = self._format_fn(
                    self, "keypoints", self._keypoints_raw
                )
            else:
                self._keypoints = Keypoints(np.zeros((0, 0, 3)), self.orig_shape)
        return self._keypoints

    @property
    def masks(self) -> Masks:
        """Lazily processes and returns masks."""
        if self._masks is None:
            if isinstance(self._masks_raw, Masks):
                self._masks = self._masks_raw
            elif self._format_fn and self._masks_raw is not None:
                self._masks = self._format_fn(self, "masks", self._masks_raw)
            else:
                self._masks = Masks(np.zeros((0, 0, 0), dtype=bool), self.orig_shape)
        return self._masks

    @property
    def orig_img(self) -> np.ndarray:
        """Lazily loads the original image if not already present."""
        if self._orig_img is None:
            self._orig_img = cv2.imread(self.path)
            if self._orig_img is None:
                # Provide a more descriptive error including file existence check
                if not Path(self.path).exists():
                    raise FileNotFoundError(
                        f"Inference result image path does not exist: {self.path}"
                    )
                raise RuntimeError(
                    f"OpenCV failed to decode image at: {self.path} (format error?)"
                )
        return self._orig_img

    def plot(self, line_width: int = 2, font_size: float = 0.5) -> np.ndarray:
        """Plots boxes, masks, and keypoints on the original image."""
        img = self.orig_img.copy()

        # 1. Draw Masks
        for seg in self.masks.xy:
            if seg.size > 0:
                cv2.polylines(
                    img,
                    [seg.astype(np.int32)],
                    True,
                    (0, 255, 0),
                    thickness=line_width,
                )

        # 2. Draw Boxes
        for i in range(len(self.boxes)):
            box = self.boxes.xyxy[i].astype(np.int32)
            conf = self.boxes.conf[i]
            cls_id = int(self.boxes.cls[i])
            name = self.names.get(cls_id, str(cls_id))

            label = f"{name} {conf:.2f}"
            cv2.rectangle(
                img,
                (box[0], box[1]),
                (box[2], box[3]),
                (255, 0, 0),
                line_width,
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
