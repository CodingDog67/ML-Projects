# two options one with augmentations and one without 
from torchvision import transforms
from torchvision.datasets import ImageFolder

import torch
import os 
import shutil
from glob import glob
import numpy as np

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

def get_data_loader(data_set, batchsize, shuffle):
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size = batchsize,
        shuffle = shuffle
    )

    return data_loader


def get_dataset(path:str, batchsize:int):
    train_dataset = ImageFolder(
        path + 'train',
        transform = no_augmentation_transforms
    )

    test_dataset = ImageFolder(
        path + 'test', 
        transform = no_augmentation_transforms
    )

    train_loader = get_data_loader(train_dataset, batchsize, True)
    test_loader = get_data_loader(test_dataset, batchsize, False)
 
    return train_dataset, test_dataset, train_loader, test_loader


# Transform the data into a flat array of features to be encoded
def get_dataset_noAug(dataset, data_loader, D, model):
  
    Ndata = len(dataset)

    x_data = np.zeros((Ndata, D))
    y_data = np.zeros((Ndata, 1))

    x_data, y_data = populate(data_loader, model, x_data, y_data)

    return x_data, y_data


def populate(data_loader, model, x_data, y_data):
    i = 0
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs = inputs.to(device)
        
            output = model(inputs)

            # size of batch can be less than batch_size
            bz = len(output)

            #assign to x_data and y_data
            x_data[i:i + bz] = output.cpu().detach().numpy()
            y_data[i:i + bz] = targets.view(-1, 1).numpy()

            i += bz

    return x_data, y_data
            