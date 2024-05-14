## This file load the data, train a model of choice, then save and validate


import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim
from model_set import *
import pickle

## Load data

# Define the transformations to resize the images
transform = transforms.Compose([
    transforms.Resize((255, 255)),
    transforms.ToTensor()
])

# Set the path to your dataset folder
dataset_folder = "images"

# Load the dataset from the folder
dataset = ImageFolder(root=dataset_folder, transform=transform)
print ("Class definition is saved")
print (dataset.class_to_idx)

# Save the dictionary to a file
with open('class_def.pkl', 'wb') as file:
    pickle.dump(dataset.class_to_idx, file)


# Define the size of the train set (80%)
train_size = int(0.8 * len(dataset))

# Define the size of the validation set (20%)
val_size = len(dataset) - train_size

# Split the dataset into train and validation sets
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Define the data loaders for training and validation sets
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Print the number of samples in each set
print(f"Number of samples in the training set: {len(train_dataset)}")
print(f"Number of samples in the validation set: {len(val_dataset)}")


# Create an instance of the SimpleCNN model
model = SimpleCNN()

# Print the model architecture
# print(model)


# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Number of training epochs
num_epochs = 8

# Training loop
for epoch in range(num_epochs):
    # Set model to training mode
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        
        # Calculate the loss
        loss = criterion(outputs, labels)
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Accumulate the loss
        running_loss += loss.item() * inputs.size(0)

    # Print average loss for the epoch
    epoch_loss = running_loss / len(train_dataset)
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}")

torch.save(model.state_dict(), 'model1.pth')


## Validation, no need
correct = 0
total = 0
# Set model to evaluation mode
model.eval()
with torch.no_grad():
    for inputs, labels in val_loader:
        outputs = model(inputs)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f"Accuracy on the test set: {accuracy:.2%}")
