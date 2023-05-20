import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import confusion_matrix
import numpy as np
import os.path 
from train import train

from plotting import plot_confusion_matrix, plot_train_test_acc, plot_train_test_loss
from cifar_cnn_model import cnn_model_cifar

""" Step up from FashionMNist, adding augmentation and batchnorm"""

# Augmentation
transformer_train = torchvision.transforms.Compose([
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomAffine(0, translate=(0.1, 0.1)),
    transforms.ToTensor(),
])

# note that this is a numpy array unlike fashionmnist, and targets are a list
train_dataset = torchvision.datasets.CIFAR10(
    root='./Fashion and Cifar',
    train = True, 
    transform=transformer_train, 
    download=False
)
train_dataset_default = torchvision.datasets.CIFAR10(
    root='./Fashion and Cifar',
    train = True, 
    transform=transforms.ToTensor(), 
    download=False
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./Fashion and Cifar',
    train = False, 
    transform=transforms.ToTensor(), 
    download=False
)

print('Datashape: ', train_dataset.data.shape)

num_classes = len(set(train_dataset.targets))
print("number of classes: ", num_classes)

batchsize = 256

train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = batchsize, shuffle = True)

train_loader_default = torch.utils.data.DataLoader(dataset = train_dataset_default, batch_size = batchsize, shuffle = False) 

test_loader = torch.utils.data.DataLoader(dataset = test_dataset, batch_size = batchsize, shuffle = False)

model = cnn_model_cifar(num_classes)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
model.to(device)

loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

if not os.path.isfile('.\\Fashion and Cifar\\trained_model_cifar.pt'):
    train_losses, test_losses, train_accs, test_accs = train(model, loss_func, optimizer, train_loader, test_loader, epochs=60, save_path='.\\Fashion and Cifar\\trained_model_cifar.pt')

    plot_train_test_loss(train_losses, test_losses)
    plot_train_test_acc(train_accs, test_accs)

else:  
    model.load_state_dict(torch.load('.\\Fashion and Cifar\\trained_model.pt'))
    model.eval()

# get all predictions in an array and plot confusion matrix
x_test = test_dataset.data.numpy()
y_test = test_dataset.targets.numpy()
p_test = np.array([])

for inputs, targets in test_loader:
  # move data to GPU
  inputs, targets = inputs.to(device), targets.to(device)

  # Forward pass
  outputs = model(inputs)

  # Get prediction
  _, predictions = torch.max(outputs, 1)
  
  # update p_test
  p_test = np.concatenate((p_test, predictions.cpu().numpy()))

cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))