import segmentation_models_pytorch as smp
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

def get_model(num_classes=4):
    # return smp.PSPNet(
    return smp.DeepLabV3(
    # return smp.Unet(

        encoder_name="resnet34",
        encoder_weights="imagenet",
        in_channels=3,
        classes=num_classes
    )