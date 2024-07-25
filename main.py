import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import dataset
import numpy as np
import architecture

# Constants
Batch_size = 64
epochs = 20
lr = 0.001
val_split = 0.2

# Load dataset
image_dir = f"{os.getcwd()}/training_data"
Dataset = dataset.ImagesDataset(image_dir, width=100, height=100, dtype=np.float32)

train_size = int((1 - val_split) * len(Dataset))
val_size = len(Dataset) - train_size
train_dataset, val_dataset = random_split(Dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=Batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=Batch_size, shuffle=False)

# Initialize the model
model = architecture.MyCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

# Training loop
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels, _, _ in train_loader:
        images = images.to(torch.float32)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch + 1}/{epochs}], Loss: {running_loss / len(train_loader):.4f}")

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels, _, _ in val_loader:
            images = images.to(torch.float32)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print(f"Validation Loss: {val_loss / len(val_loader):.4f}, Accuracy: {correct / total:.4f}")

# Save model
torch.save(model.state_dict(), "model.pth")
print("Model saved as model.pth")