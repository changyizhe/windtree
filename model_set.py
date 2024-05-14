## A file define all models, each model shall be a class

import os
import torch
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
import torch.optim as optim


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(SimpleCNN, self).__init__()
        # Define the convolutional layer
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1)
        # Define the max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=3, stride=3)
        # Define the flattening layer
        self.flatten = nn.Flatten()
        # Define the fully connected (dense) layers
        self.fc1 = nn.Linear(16 * 85 * 85, 100)  # 85x85 is the output size after pooling
        self.fc2 = nn.Linear(100, num_classes)
        # Define the activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        # Forward pass through the network
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x