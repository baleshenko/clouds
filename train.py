import numpy as np
import torch
import torch.nn as nn
import tqdm

from config import load_config
from data import build_dataloaders, mixup_batch
from metrics import EpochMetrics, MetricsTracker
from model import CloudClassifier


def main() -> None:
    ...


if __name__ == "__main__":
    main()