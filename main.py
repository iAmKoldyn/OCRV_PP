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
        mask = cv2.imread(mask_path, 0) 
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST) 

        mask = np.where(mask == 113, 1, mask)
        mask = np.where(mask == 169, 2, mask)
        mask = np.where(mask == 227, 2, mask)

        unique_values = np.unique(mask)
        print(f"Unique mask values: {unique_values}")


        mask = torch.from_numpy(mask).long()

        if self.transform is not None:
            image = self.transform(image)

        return image, mask

def get_model(num_classes=3):
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=num_classes
    )
    return model

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


def iou_score(output, target):
    output = torch.sigmoid(output)
    output = torch.argmax(output, dim=1)
    output = output.view(-1)
    target = target.view(-1)

    intersection = (output == target).float().sum()
    union = output.numel()

    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.item()


def dice_score(output, target):
    output = torch.sigmoid(output)
    output = torch.argmax(output, dim=1)
    output = output.view(-1)
    target = target.view(-1)

    intersection = (output == target).float().sum()
    dice = (2. * intersection + 1e-6) / (output.numel() + target.numel() + 1e-6)
    return dice.item()


def visualize(model, data_loader, num_images=5):
    model.eval()
    images, _ = next(iter(data_loader))
    with torch.no_grad():
        preds = model(images)
    preds = torch.argmax(preds, dim=1)
    images = images.cpu().numpy()
    preds = preds.cpu().numpy()
    fig, ax = plt.subplots(nrows=num_images, ncols=2, figsize=(10, num_images * 5))
    for i in range(num_images):
        ax[i, 0].imshow(np.transpose(images[i], (1, 2, 0)))
        ax[i, 1].imshow(preds[i], cmap='gray')
        ax[i, 0].set_title("Original Image")
        ax[i, 1].set_title("Predicted Mask")
        ax[i, 0].axis('off')
        ax[i, 1].axis('off')
    plt.show()

def main():
    dataset_root = 'segmentation_labeled_dataset'
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

    model = get_model(num_classes=3)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    num_epochs = 25
    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = 0
        iou_total = 0
        dice_total = 0
        count = 0
        for images, masks in val_loader:
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_loss += loss.item()
            iou_total += iou_score(outputs, masks)
            dice_total += dice_score(outputs, masks)
            count += 1
        val_loss /= count
        avg_iou = iou_total / count
        avg_dice = dice_total / count
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}, IoU: {avg_iou}, Dice: {avg_dice}")

    visualize(model, val_loader)

if __name__ == "__main__":
    main()