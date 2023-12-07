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


def get_model(num_classes=4):
    # model = smp.PSPNet(
    # model = smp.DeepLabV3(
    model = smp.Unet(
        encoder_name="resnet34", 
        encoder_weights="imagenet", 
        in_channels=3, 
        classes=num_classes)

    return model


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

    model = get_model(num_classes=4)
    criterion = nn.CrossEntropyLoss()
    # optimizer = optim.Adam(model.parameters(), lr=1e-3)
    # optimizer = optim.Adam(model.parameters(), lr=1e-4)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    # optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)
    # optimizer = optim.RMSprop(model.parameters(), lr=1e-4)


    best_val_loss = float('inf')
    best_model_path = ""
    num_epochs = 25

    for epoch in range(num_epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss = validate(model, val_loader, criterion)
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Val Loss: {val_loss}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_path = f'model_checkpoint_epoch_{epoch}.pth'
            torch.save(model.state_dict(), best_model_path)
            print(f"Checkpoint saved for epoch {epoch} with validation loss: {val_loss}")

    print("Training complete.")

    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))

    visualize_results = True
    if visualize_results:
        visualize(model, val_loader)

if __name__ == "__main__":
    main()