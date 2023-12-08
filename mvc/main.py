import torch
import torch.nn as nn
from dataset.dataset import FootballFieldDataset
from model.model import get_model
from controller.controller import train_model
from view.view import visualize
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
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


def main():
    dataset_root = 'segmentation_labeled_dataset_V2'
    transform = transforms.Compose([
        transforms.Resize((256, 256)), 
        transforms.ToTensor()
    ])

    train_dataset = FootballFieldDataset(
        image_dir=os.path.join(dataset_root, 'train', 'JPEGImages'),
        mask_dir=os.path.join(dataset_root, 'train', 'SegmentationClass'),
        transform=transform
    )
    val_dataset = FootballFieldDataset(
        image_dir=os.path.join(dataset_root, 'val', 'JPEGImages'),
        mask_dir=os.path.join(dataset_root, 'val', 'SegmentationClass'),
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

    model = get_model(num_classes=4)
    criterion = nn.CrossEntropyLoss()

    best_model_path = train_model(model, train_loader, val_loader, criterion)

    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))

    visualize_results = True
    if visualize_results:
        visualize(model, val_loader)

if __name__ == "__main__":
    main()
