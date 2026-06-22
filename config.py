import yaml
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class TrainConfig(BaseModel):
    img: int = Field(224, ge=32, le=1024, description="Square crop")
    batch: int = Field(16, ge=1, le=512)
    lr: float = Field(1e-4, gt=0, lt=1.0)
    weight_decay: float = Field(5e-4, ge=0)
    epochs: int = Field(50, ge=1, le=1000)
    patience: int = Field(10, ge=1, description="Early-stopping")
    label_smoothing: float = Field(0.1, ge=0.0, lt=0.5)
    mixup_alpha: float = Field(.2, ge=0.0, description="0 = disabled")
    num_workers: int = Field(4, ge=0)

    @field_validator("img")
    @classmethod
    def must_be_divisible_by_32(cls, i: int) -> int:
        if i % 32 != 0:
            raise ValueError("img must be divisible by 32")
        return i
    

class ModelConfig(BaseModel):
    backbone: Literal[
        "efficientnet_b0", "efficientnet_b2", "efficientnet_b4",
        "resnet18", "resnet34", "resnet50",
    ] = "efficientnet_b0"
    hidden: int = Field(256, ge=64)
    dropout: float = Field(.4, ge=.0, lt=1.)
    freeze_backbone: bool = True
    unfreeze_after_epoch: int = Field(
        5,
        ge=0,
        description="backbone after N epochs")
    

class PathsConfig(BaseModel):
    train: Path = Path("data/train")
    val: Path = Path("data/val")
    checkpoint: Path = Path("checkpoints/best_model.pth")
    records: Path = Path("records")

    @model_validator(mode="after")
    def ensure_log_dir(self) -> "PathsConfig":
        self.records.mkdir(parents=True, exist_ok=True)
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        return self


class LogfireConfig(BaseModel):
    enabled: bool = True
    project_name: str = "clouds"
    service_name: str = "train"


class AppConfig(BaseModel):
    train: TrainConfig = Field(default_factory=TrainConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    logfire: LogfireConfig = Field(default_factory=LogfireConfig)

    device: str = "auto"
    seed: int = 42

    classes: list[str] = Field(
        default_factory=lambda: ["cumulus", "cumulonimbus"]
    )

    @field_validator("device", mode="before")
    @classmethod
    def resolve_device(cls, value: str) -> str:
        return "cuda" if value == "auto" else value
    


def load_config(path: str | Path = "config.yaml") -> AppConfig:
    """Load & validate config from a YAML file"""

    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return AppConfig.model_validate(raw)
