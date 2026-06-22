import torch
import torch.nn as nn

from torchvision import models
from torchvision.models import (
    EfficientNet_B0_Weights, EfficientNet_B2_Weights, EfficientNet_B4_Weights,
    ResNet18_Weights, ResNet34_Weights, ResNet50_Weights)

from config import ModelConfig


_EFFICIENTNET_REGISTRY: dict[str, tuple] = {
    "efficientnet_b0": (models.efficientnet_b0, EfficientNet_B0_Weights.IMAGENET1K_V1, 1280),
    "efficientnet_b2": (models.efficientnet_b2, EfficientNet_B2_Weights.IMAGENET1K_V1, 1408),
    "efficientnet_b4": (models.efficientnet_b4, EfficientNet_B4_Weights.IMAGENET1K_V1, 1792),
}

_RESNET_REGISTRY: dict[str, tuple] = {
    "resnet18": (models.resnet18, ResNet18_Weights.IMAGENET1K_V1, 512),
    "resnet34": (models.resnet34, ResNet34_Weights.IMAGENET1K_V1, 512),
    "resnet50": (models.resnet50, ResNet50_Weights.IMAGENET1K_V1, 2048),
}


class CloudClassifier(nn.Module):
    def __init__(self, config: ModelConfig, count_classes: int) -> None:
        super().__init__()