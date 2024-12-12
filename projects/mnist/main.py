import torch
import torch.nn as nn
import torch.optim as optim
from utils import setup_device, download_dataset
from torch.utils.data import DataLoader
from models.simple import SimpleNN
import datetime

"""
TODO:

1. find out the data type and values of train_dataset, test_dataset
2. find out the dimensions of: (data, target)
3. find the meaning of .to() function in data.to(device)
4. find out the data type, dimensions, and values of output = model(data)
"""

# train function
def train(model, device, train_loader, criterion, optimizer, epochs=5) -> None:
    model.train()  # Set the model to training mode
    for epoch in range(epochs):
        running_loss = 0.0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass: Compute the model output
            output = model(data)
            # Compute the loss
            loss = criterion(output, target)
            # Backward pass: Compute gradients
            loss.backward()
            # Update the weights
            optimizer.step()
            running_loss += loss.item()
            # Print progress every 100 batches
            if batch_idx % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")

        print(f"Epoch {epoch+1} completed, Avg. Loss: {running_loss / len(train_loader):.4f}")

# test function
def test(model, device, test_loader) -> None:
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient calculation for evaluation
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)  # Get the index of the max log-probability
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f"Test Accuracy: {100 * correct / total:.2f}%")

if __name__ == "__main__":
    device = setup_device()
    data_dir = "./data/mnist"
    train_dataset, test_dataset = download_dataset(data_dir)
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)
    # setup the model
    model = SimpleNN().to(device)
    # define other objects
    # Loss function: Cross-Entropy Loss (for classification tasks)
    criterion = nn.CrossEntropyLoss()
    # Optimizer: Stochastic Gradient Descent (SGD) with learning rate of 0.01
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    start_time = datetime.datetime.now()
    # Train the model for 5 epochs
    train(model, device, train_loader, criterion, optimizer, epochs=5)
    # Test the model after training
    test(model, device, test_loader)
    end_time = datetime.datetime.now()
    elapsed_time = end_time - start_time
    print(f"Time elapsed: {elapsed_time}")