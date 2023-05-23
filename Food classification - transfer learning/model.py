import torch.nn as nn
from torchvision import models


class model_augmentation(nn.Module):
     # using a pretrained model backbone
    def __init__(self, vgg):
        super(model_augmentation, self).__init__()
        self.model = vgg
        self.classifier =  nn.Sequential(nn.Linear(vgg.classifier[0].in_features, 512), 
                                          nn.ReLU(), 
                                          nn.Linear(512, 2))
    def forward(self, x):
        x = self.model.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
    def freeze_param(self):
        for param in self.model.parameters():
            param.requires_grad = False
    

class model_NoAug(nn.Module):
    def __init__(self, vgg):
        super(model_NoAug, self).__init__()
        self.vgg = vgg

    def forward(self, X):
        x = self.vgg.features(X)
        x = self.vgg.avgpool(x)
        x = x.view(x.size(0), -1)
        return x
