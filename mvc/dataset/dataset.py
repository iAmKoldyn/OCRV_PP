from torch.utils.data import Dataset
import os
import cv2
import numpy as np
import torch
from torchvision.transforms import functional as F
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


class FootballFieldDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx])
        image = cv2.imread(img_path)
        mask = cv2.imread(mask_path, 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = F.to_pil_image(image)
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask == 113, 1, mask)
        mask = np.where(mask == 169, 2, mask)
        mask = np.where(mask == 227, 2, mask)
        mask = torch.from_numpy(mask).long()
        if self.transform is not None:
            image = self.transform(image)
        return image, mask
