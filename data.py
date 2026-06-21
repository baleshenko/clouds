import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from config import TrainConfig


_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]



def build_transforms(config: TrainConfig) -> tuple[transforms.Compose, transforms.Compose]: 
    """ Data Augmentation """
    ...


def dataloaders(train: Path, val: Path, config: TrainConfig) -> tuple[DataLoader, DataLoader, list[str]]:  
    """ Create Dataloaders """ 
    ...