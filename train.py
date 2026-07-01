import numpy as np
import random
import torch
import torch.nn as nn
import tqdm

from config import load_config
from data import build_dataloaders, mixup_batch
from metrics import EpochMetrics, MetricsTracker
from model import CloudClassifier



def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False


def train_one_epoch(
    model: nn.Module,
    loader,
    optim: torch.optim.Optimizer,
    device: str,
    mixup_alpha: float,
    num_classes: int,
) -> float:
    
    """
    Trains the model for one epoch using MixUp augmentation.

    Args:
        model (nn.Module): Neural network model.
        loader: DataLoader providing training batches.
        optim (torch.optim.Optimizer): Optimizer for parameter updates.
        device (str): Device to run computation on ('cuda').
        mixup_alpha (float): MixUp interpolation strength parameter.
        num_classes (int): Number of target classes.

    Returns:
        float: Average training loss over the epoch.
    """
    
    model.train()
    total_loss = 0.

    for images, labels in tqdm.tqdm(loader, 
                                    desc="  train", 
                                    leave=False, 
                                    colour="yellow"):
        
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        images, soft_labels = mixup_batch(images, labels, mixup_alpha, num_classes)

        optim.zero_grad(set_to_none=True)
        logits = model(images)

        log_probs = torch.log_softmax(logits, dim=1)
        loss = -(soft_labels * log_probs).sum(dim=1).mean()

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.)
        optim.step()

        total_loss += loss.item()

    return total_loss / max(len(loader), 1)



def main() -> None:
    ...


if __name__ == "__main__":
    main()