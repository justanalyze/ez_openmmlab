from pathlib import Path
from typing import List, Optional, Tuple, Union

import tomli
import tomli_w
from pydantic import BaseModel, ConfigDict, Field, computed_field

from ez_openmmlab.core.schema.models import ModelName


# --- Pydantic Models for Validation ---
class ModelSection(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    name: ModelName = ModelName.RTM_DET_TINY
    num_classes: Optional[int] = Field(None, gt=0)
    num_keypoints: Optional[int] = Field(None, gt=0)
    load_from: Optional[str] = None
    base_config_path: Optional[str] = None


class DataSection(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    root: str
    dataset_name: Optional[str] = None
    train_ann: str = "annotations/instances_train2017.json"
    train_img: str = "train2017/"
    val_ann: str = "annotations/instances_val2017.json"
    val_img: str = "val2017/"
    test_ann: Optional[str] = None
    test_img: Optional[str] = None
    classes: Optional[List[str]] = None
    metainfo: Optional[dict] = None

    @computed_field
    @property
    def train_ann_path(self) -> str:
        return str(Path(self.root) / self.train_ann)

    @computed_field
    @property
    def train_img_path(self) -> str:
        return str(Path(self.root) / self.train_img)

    @computed_field
    @property
    def val_ann_path(self) -> str:
        return str(Path(self.root) / self.val_ann)

    @computed_field
    @property
    def val_img_path(self) -> str:
        return str(Path(self.root) / self.val_img)

    @computed_field
    @property
    def test_ann_path(self) -> Optional[str]:
        return str(Path(self.root) / self.test_ann) if self.test_ann else None

    @computed_field
    @property
    def test_img_path(self) -> Optional[str]:
        return str(Path(self.root) / self.test_img) if self.test_img else None


class TrainingSection(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    epochs: int = Field(100, gt=0)
    batch_size: int = Field(8, gt=0)
    num_workers: int = Field(2, ge=0, description="Number of dataloader workers")
    learning_rate: float = Field(0.001, gt=0.0)
    weight_decay: Optional[float] = Field(None, ge=0.0)
    device: str = "cuda"
    work_dir: str = "./runs/train"
    log_level: str = "INFO"
    enable_tensorboard: bool = Field(True, description="Enable TensorBoard logging")
    amp: bool = True
    evaluator_metric: Optional[Union[str, dict, List[Union[str, dict]]]] = None
    resume: Union[bool, str] = False
    stage2_num_epochs: int = Field(
        20, ge=0, description="Number of epochs for stage 2 training pipeline"
    )


class AugmentationSection(BaseModel):
    """Configuration for data augmentation parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    scale_factor: Optional[Union[float, Tuple[float, float], List[float]]] = Field(
        None, description="Scaling factor or range for augmentations."
    )
    rotate_factor: Optional[float] = Field(
        None, ge=0.0, description="Rotation factor for augmentations (degrees)."
    )
    random_flip_prob: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Probability of random horizontal flip. Default 0.5 if relevant transform is present.",
    )


class UserConfig(BaseModel):
    """The master schema for config.toml."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    model: ModelSection
    data: DataSection
    training: TrainingSection
    augments: AugmentationSection = Field(default_factory=AugmentationSection)


# --- Utilities ---
def load_user_config(path: Path) -> UserConfig:
    """Reads a TOML file and validates it against the schema."""
    with open(path, "rb") as f:
        data = tomli.load(f)
    return UserConfig(**data)


def save_user_config(config: UserConfig, path: Path) -> None:
    """Writes the configuration to a TOML file."""
    # Convert Pydantic model to dict, filtering None to keep it clean
    data = config.model_dump(exclude_none=True)

    # Write to file
    with open(path, "wb") as f:
        tomli_w.dump(data, f)

