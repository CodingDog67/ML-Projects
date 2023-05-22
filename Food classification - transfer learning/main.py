# example of inference learning on a pretrained vgg16 model 
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, models, transforms

import numpy as np
import imageio
import matplotlib.pyplot as plt
from dataset import get_dataset
from datetime import datetime


train_dataset, test_dataset, train_loader, test_loader = get_dataset(path = ".\\Food classification - transfer learning\\data\\",batchsize=32)

# using a pretrained model backbone
model = models.vgg16(pretrained=True)

# Freeze the vgg weights
for param in model.parameters():
    param.requires_grad = False

#model.summary()

print(model.classifier)

# replace the classifier 
in_features = model.classifier[0].in_features

model.classifier = nn(in_features, 2)  # binary 