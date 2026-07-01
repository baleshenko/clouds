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
    def __init__(self, config: ModelConfig, 
                 count_classes: int) -> None:
        super().__init__()
        self.config = config
        self._build(count_classes)
    
    def _build(self, count_classes: int) -> None:        
        if self.config.backbone in _EFFICIENTNET_REGISTRY:
            factory, weights, in_features = _EFFICIENTNET_REGISTRY[self.config.backbone]
            
            base = factory(weights=weights)
            base.classifier = nn.Sequential(
                nn.Dropout(p=self.config.dropout, inplace=True),
                nn.Linear(in_features, self.config.hidden),
                nn.SiLU(),
                nn.Dropout(p=self.config.dropout / 2),
                nn.Linear(self.config.hidden, count_classes))
            
            self._backbone_params = list(base.features.parameters())
            self._head_params = list(base.classifier.parameters())

        elif self.config.backbone in _RESNET_REGISTRY:
            factory, weights, in_features = _RESNET_REGISTRY[self.config.backbone]
            base = factory(weights=weights)
            base.fc = nn.Sequential(
                nn.Linear(in_features, self.config.hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(self.config.dropout),
                nn.Linear(self.config.hidden, count_classes),
            )
            self._backbone_params = [
                i for name, i in base.named_parameters()
                if not name.startswith("fc")
            ]
            self._head_params = list(base.fc.parameters())

        else:
            raise ValueError(f"Unknown backbone: {self.config.backbone}")

        self.base = base

        if self.config.backbone_train:
            self.backbone_train()


    def backbone_train(self) -> None:
        for i in self._backbone_params:
            i.requires_grad = False

    def release_backbone(self) -> None:
        for i in self._backbone_params:
            i.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x)


    def head_parameters(self):
        return self._head_params

    def all_parameters(self):
        return list(self.parameters())

    def count_parameters(self) -> dict[str, int]:
        total = sum(i.numel() for i in self.parameters())
        train_ = sum(i.numel() for i in self.parameters() if i.requires_grad)
        return {"total": total, "train_": train_, "block": total - train_}
