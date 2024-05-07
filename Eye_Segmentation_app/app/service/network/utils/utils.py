import os
import torch
import torchvision

from torch.utils.data import DataLoader

from network.processing.dataset import FaceDataset

SAVE_DIR = "Saved_Images"

def save_checkpoint(state, filename="checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])

def get_loaders(
    train_dir,
    train_mask_dir,
    val_dir,
    val_mask_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True
):
    train_data = FaceDataset(image_dir=train_dir, mask_dir=train_mask_dir, transform = train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    val_data = FaceDataset(image_dir=val_dir, mask_dir=val_mask_dir, transform = val_transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, num_workers=num_workers, pin_memory=pin_memory, shuffle=False)

    return train_loader, val_loader

def save_predictions_as_imgs(loader, model, folder = SAVE_DIR, device="cuda"):
    if not os.path.exists(folder):
      os.mkdir(folder)

    model.eval()

    for i , (x, y) in enumerate(loader):
        x = x.to(device)
        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            torchvision.utils.save_image(preds, f"{folder}/pred_{i}.jpg")
            torchvision.utils.save_image(y, f"{folder}/truth_{i}.jpg")