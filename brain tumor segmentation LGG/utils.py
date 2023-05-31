import torch
import torchvision
from dataset import lgg_mri_dataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename = "newest_checkpoint.pth.tar"):
    print("Saving Checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("Loading model")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(train_dir, train_maskdir, valdir, val_maskdir, batchsize, 
                train_transform, val_transform, num_workers=4, pin_memory=True):
    
    train_ds = lgg_mri_dataset(train_dir, train_maskdir, train_transform)
    test_ds = lgg_mri_dataset(valdir, val_maskdir, val_transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=batchsize,
        #num_workers=num_workers,
        #pin_memory=pin_memory,
        shuffle=True
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=batchsize,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False
    )

    return train_loader, test_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0

    model.eval()

    with torch.no_grad():
        for inputs, targets, in loader:
            inputs = inputs.to(device)
            targets = targets.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(inputs))
            preds = (preds > 0.5).float()

            num_correct += (preds == targets).sum()
            num_pixels += torch.numel(preds) # num of elements
            dice_score += (2*(preds * targets).sum()) / ((preds + targets).sum() + 1e-8)  # binary dice score


        print(f"Correctly segmented {num_correct} / {num_pixels} accuracy is {num_correct/num_pixels*100:.2f}")
        print(f"Dice score: {dice_score/len(loader)}")

        model.train()