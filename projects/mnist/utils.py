import torch
import os
from torchvision import datasets, transforms
from typing import Tuple

transform = transforms.Compose([
    transforms.ToTensor(), # convert the values to [0, 1]
    transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1], new_val = (old_val - mean) / std
])

def setup_device() -> torch.device:
    # Check if MPS is available (for Apple Silicon devices)
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    # Check if CUDA is available (for NVIDIA GPUs)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    # Fallback to CPU if neither is available
    else:
        device = torch.device("cpu")
    return device

def download_dataset(data_dir: str) -> Tuple[datasets.MNIST, datasets.MNIST]:
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    train_dataset = datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
    # print(len(train_dataset))
    print(type(train_dataset))
    print(type(test_dataset))
    return train_dataset, test_dataset