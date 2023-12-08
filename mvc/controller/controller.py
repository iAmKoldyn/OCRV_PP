import torch
from torch.optim import Adam
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import segmentation_models_pytorch as smp
import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision.transforms import functional as F

def train_one_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    for images, masks in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(dataloader)

def validate(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, masks in dataloader:
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
    return total_loss / len(dataloader)

def train_model(model, train_loader, val_loader, criterion, num_epochs=25):
    optimizer = Adam(model.parameters(), lr=1e-4)
    best_val_loss = float('inf')
    best_model_path = ""

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = f'model_checkpoint_epoch_{epoch}.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"Checkpoint saved for epoch {epoch} with validation loss: {val_loss}")

    return best_model_path
