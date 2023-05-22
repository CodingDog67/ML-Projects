""" Example on how to use inference learning on a pretrained vgg16 model with Data augmentation or without and a fixed feature transformed set"""

import torch
import torch.nn as nn
from torchvision import models

import sys
import numpy as np
from dataset import get_dataset
from datetime import datetime

sys.path.insert(1, './Fashion and Cifar/')
import plotting
from model import model_augmentation

def main():
    _, _, train_loader, test_loader = get_dataset(path = ".\\Food classification - transfer learning\\data\\",batchsize=32)

    # using a pretrained model backbone
    model_vgg = models.vgg16(pretrained=True)

    model = model_augmentation(model_vgg)
    model.freeze_param()

    print(model)
    print(model.classifier)

    # move model to gpu 
    device = torch.device('cuda:0' if torch.cuda.is_available else 'cpu')
    print(device)
    model.to(device)

    epochs = 6

    # Loss and optimizer
    optim = torch.optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss()

    train_losses, test_losses, model = fit(model, criterion, optim, train_loader, test_loader, epochs, device)
    
    plotting.plot_train_test_loss(train_losses, test_losses)

    acc_train = get_accuracy(train_loader, model, device)
    acc_test = get_accuracy(test_loader, model, device)

    print(f"Train accuracy: {acc_train: .4f}, Test accuracy: {acc_test: .4f}")


def fit(model, criterion, optim, train_loader, test_loader, epochs, device):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):
        t_0 = datetime.now()
        train_loss = []
        test_loss = []

        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets= targets.to(device)
            
            #zero the param gradients
            optim.zero_grad()

            # forward pass
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            # backward pass and optimization
            loss.backward()
            optim.step()

            train_loss.append(loss.item())

        train_loss = np.mean(train_loss)

        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_loss.append(loss.item())
        
        test_loss = np.mean(test_loss)

        train_losses[it] = train_loss
        test_losses[it] = test_loss 
    
        dt = datetime.now() - t_0
        print(f'Epoch{it+1}/{epochs}, Train Loss: {train_loss: .4f}, \
              Test Loss: {test_loss:.4f}, Duration: {dt}')
    
    return train_losses, test_losses, model
    
def get_accuracy(loader, model, device):
    n_correct = 0
    n_total = 0
    
    for inputs, targets in loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = model(inputs)
        _, predictions = torch.max(outputs, 1)

        n_correct += (predictions == targets).sum().item()
        n_total += targets.shape[0]

    acc = n_correct/n_total

    return acc

if __name__ == '__main__':
    main()