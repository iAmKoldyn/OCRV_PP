import matplotlib.pyplot as plt
import numpy as np
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

def post_process_mask(mask):
    mask = mask.astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel, iterations=2)
    return closing


def visualize(model, data_loader, num_images=5):
    model.eval()
    images, _ = next(iter(data_loader))
    with torch.no_grad():
        preds = model(images)
    preds = torch.argmax(preds, dim=1).cpu().numpy()
    images = images.cpu().numpy()

    num_images = min(num_images, len(images))
    fig, ax = plt.subplots(nrows=num_images, ncols=3, figsize=(15, num_images * 5))
    for i in range(num_images):
        processed_mask = post_process_mask(preds[i])
        ax[i, 0].imshow(np.transpose(images[i], (1, 2, 0)))
        ax[i, 1].imshow(preds[i], cmap='gray')
        ax[i, 2].imshow(processed_mask, cmap='gray')
        ax[i, 0].set_title("Original Image")
        ax[i, 1].set_title("Predicted Mask")
        ax[i, 2].set_title("Post Processed Mask")
        ax[i, 0].axis('off')
        ax[i, 1].axis('off')
        ax[i, 2].axis('off')
    plt.show()
