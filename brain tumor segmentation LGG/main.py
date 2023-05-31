import torch
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import albumentations as A
from torchvision import transforms
import yaml

import numpy as np
from segmentation_models_pytorch.encoders import get_preprocessing_fn
from model import UNET
from tqdm import tqdm
from albumentations.pytorch import ToTensorV2
from utils import get_loaders, check_accuracy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# temp hyperparameters to be read from yaml later
TRAIN_IMG_DIR = '.\\brain tumor segmentation LGG\\data\\train_images'
TRAIN_MASK_DIR = '.\\brain tumor segmentation LGG\\data\\train_mask'
VAL_IMG_DIR = '.\\brain tumor segmentation LGG\\data\\val_images'
VAL_MASK_DIR = '.\\brain tumor segmentation LGG\\data\\val_mask'

IMG_HEIGHT= 256
IMG_WIDTH = 256
BATCHSIZE = 16

LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
NUM_WORKERS = 2

PIN_MEMORY = True
LOAD_MODEL = False

# will do one epoch of training
def train(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device)
        targets = targets.float().unsqueeze(1).to(device) # add channel dimension

        # foward pass float 16 training
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item)

def main():
    train_transform = transforms.Compose([
        transforms.ToTensor(),
        #transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),  # EXPECTs TENSORIMAGE >_>, totensor first 
        ]
    )

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]),
        ]
    )

    model = UNET(in_channels=3, out_channels=1).to(device)
    loss_fn = nn.BCEWithLogitsLoss() # since we dont use sigmoid on UNET vanilla, cross entropy loss if multiclass
    optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR, 
        TRAIN_MASK_DIR, 
        VAL_IMG_DIR, 
        VAL_MASK_DIR, 
        BATCHSIZE, 
        train_transform, 
        test_transform
        )

    scaler = torch.cuda.amp.GradScaler()

    for epochs in range(NUM_EPOCHS):
        train(train_loader, model, optimizer, loss_fn, scaler)

        # save model

        check_accuracy(val_loader, model, device = device)

if __name__ == "__main__":
    main()