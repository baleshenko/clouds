import torch
import random
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
    frame = config.img_size

    train_tfm = transforms.Compose([
        transforms.Resize((frame, frame)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(p=.1),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=.3, 
                               contrast=.3, 
                               saturation=.3, 
                               hue=.1),
        transforms.RandomGrayscale(p=.05),

        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
        transforms.RandomErasing(p=.1)
    ])

    val_tfm = transforms.Compose([
        transforms.Resize((frame, frame)),
        transforms.ToTensor(),
        transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD),
    ])

    return train_tfm, val_tfm


def mixup_batch(
    images: torch.Tensor,
    labels: torch.Tensor,
    alpha: float,
    num_classes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    
    if alpha <= .0:
        one_hot = torch.zeros(labels.size(0), 
                              num_classes, 
                              device=labels.device)
        one_hot.scatter_(1, labels.unsqueeze(1), 1.)
        return images, one_hot

    lam = float(np.random.beta(alpha, alpha))
    batch = images.size(0)
    
    idx = list(range(batch))
    random.shuffle(idx)
    idx = torch.tensor(idx, device=images.device)

    mixed = lam * images + (1 - lam) * images[idx]

    y_a = torch.zeros(batch, num_classes, device=labels.device)
    y_b = torch.zeros(batch, num_classes, device=labels.device)
    y_a.scatter_(1, labels.unsqueeze(1), 1.0)
    y_b.scatter_(1, labels[idx].unsqueeze(1), 1.0)

    soft_labels = lam * y_a + (1 - lam) * y_b
    return mixed, soft_labels



def dataloaders(train: Path, val: Path, config: TrainConfig) -> tuple[DataLoader, DataLoader, list[str]]:  
    """ Create Dataloaders """ 
    
    for d in (train, val):
        if not d.exists():
            raise FileNotFoundError(
                f"not found: {d}\n"
                "Expected:\n"
                "  data/train/cumulus/\n"
                "  data/train/cumulonimbus/\n"
                "  data/val/cumulus/\n"
                "  data/val/cumulonimbus/"
            )

    train_tfm, val_tfm = build_transforms(config)

    train_ds = datasets.ImageFolder(str(train), transform=train_tfm)
    val_ds   = datasets.ImageFolder(str(val),   transform=val_tfm)

    persistent = config.num_workers > 0

    train_loader = DataLoader(
        train_ds,
        batch=config.batch,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
        drop_last=True
    )
    val_loader = DataLoader(
        val_ds,
        batch=config.batch * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=persistent,
    )

    return train_loader, val_loader, train_ds.classes