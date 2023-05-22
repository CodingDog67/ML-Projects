# two options one with augmentations and one without 
from torchvision import transforms
from torchvision.datasets import ImageFolder
import torch
import os 
import shutil
from glob import glob

def move_img(old_path, newPath):
    list_images= glob(old_path)
    for img in list_images:
        shutil.move(img, newPath) 

#1 is food and 0 isnt 
move_img('.\\Food classification - transfer learning\\data\\FoodImage\\training\\1*.jpg',
         '.\\Food classification - transfer learning\\data\\train\\food')

move_img('.\\Food classification - transfer learning\\data\\FoodImage\\training\\1*.jpg',
         '.\\Food classification - transfer learning\\data\\train\\food')

move_img('.\\Food classification - transfer learning\\data\\FoodImage\\validation\\1*.jpg',
         '.\\Food classification - transfer learning\\data\\test\\food')

move_img('.\\Food classification - transfer learning\\data\\FoodImage\\validation\\0*.jpg',
         '.\\Food classification - transfer learning\\data\\test\\nonfood')

train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
    transforms.ColorJitter(),
    transforms.RandomRotation(degrees=15),
    transforms.RandomHorizontalFlip(),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #standardized mean and std for imageNet
])

test_transforms = transforms.Compose([
    transforms.Resize(size=256),
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) #standardized mean and std for imageNet
])

# Use this instead of augmentation transforms above to a faster classification at the cost of accuracy loss
no_augmentation_transforms = transforms.Compose([
    transforms.Resize(size=256), 
    transforms.CenterCrop(size=224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def get_dataset(path:str, batchsize:int):
    train_dataset = ImageFolder(
        path + 'train',
        transform = no_augmentation_transforms
    )

    test_dataset = ImageFolder(
        path + 'test', 
        transform = no_augmentation_transforms
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size = batchsize, 
        shuffle = True
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size = batchsize,
        shuffle = False
    )

    return train_dataset, test_dataset, train_loader, test_loader
    

