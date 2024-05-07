import torch
import torch.optim as optim

from model.Unet import myUNET
from loss.dice_loss import DiceLoss
from utils.utils import *
from processing.transforms import get_transforms
from metrics.accuracy import check_accuracy
from training.train_loop import train

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LEARNING_RATE = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 50
NUM_WORKERS = 2
IMAGE_HEIGHT = 218
IMAGE_WIDTH = 178
PIN_MEMORY = True
LOAD_MODEL = True
IMG_DIR_TRAIN = "/Portfolio/Eye_Segmentation_Project/Dataset_Faces_training"
IMG_DIR_VAL = "/Portfolio/Eye_Segmentation_Project/Dataset_Faces_validation"
MASK_DIR_TRAIN = "/Portfolio/Eye_Segmentation_Project/Dataset_Faces_Mask_training"
MASK_DIR_VAL = "/Portfolio/Eye_Segmentation_Project/Dataset_Faces_Mask_validation"
SAVE_DIR = "Saved_Images"

train_transform, validation_transform = get_transforms()

train_loader, val_loader = get_loaders(
    IMG_DIR_TRAIN,
    MASK_DIR_TRAIN,
    IMG_DIR_VAL,
    MASK_DIR_VAL,
    BATCH_SIZE,
    train_transform,
    validation_transform,
    NUM_WORKERS,
    PIN_MEMORY
)

UNET = myUNET(in_channels=3, out_channels=1).to(DEVICE)

loss_fn = DiceLoss() #Needed to force the NN to chose another strategy than putting every pixel to white
optimizer = optim.Adam(UNET.parameters(), lr=LEARNING_RATE)

scaler = torch.cuda.amp.GradScaler()

for epoch in range(NUM_EPOCHS):
    #print(train_loader)
    train(train_loader, UNET, optimizer, loss_fn, scaler)

    checkpoint =  {
        "state_dict": UNET.state_dict(),
        "optimizer": optimizer.state_dict()
    }

    save_checkpoint(checkpoint)

    check_accuracy(val_loader, UNET, device=DEVICE)

    save_predictions_as_imgs(val_loader, UNET, folder="/Saved_Images", device=DEVICE)