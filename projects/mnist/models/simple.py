import torch
import torch.nn as nn
import torch.optim as optim
from utils import setup_device, download_dataset
from torch.utils.data import DataLoader
from typing import cast

# define the structure of neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)  # Fully connected layer 1
        self.fc2 = nn.Linear(128, 64)       # Fully connected layer 2
        self.fc3 = nn.Linear(64, 10)        # Output layer for 10 classes (digits 0-9)

    # there is no restrctions about the input x when overwriting
    def forward(self, x: torch.Tensor):
        # the size -1 is inferred from other dimensions, it means the number of samples
        # in a batch. The second dimension means the size of 1-D dimension of a graph
        x = x.view(-1, 28 * 28)  # Flatten the image
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)           # No activation for the final output (applies softmax in loss)
        return x