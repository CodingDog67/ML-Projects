import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

from torch import optim
from torch.autograd import Variable
from segmentation_models_pytorch.encoders import get_preprocessing_fn

# https://smp.readthedocs.io/en/latest/quickstart.html give this a look 

# UNets double conv before self pooling 
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,stride=1,padding=1, bias=False),  # same conv, bias no neccesary due to batchnorm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3,1,1, bias=False),  # same conv, bias no neccesary due to batchnorm
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, in_channels=3, out_channels=1, features=[64,128,256,512]): # Binary needs onely 1 channel
        super(UNET, self).__init__()

        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride =2)  # floor division, take care of dimonsionality change

        # Downpart UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Bottompart UNET
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # Uppart UNET
        for feature in features[::-1]:
            self.ups.append(
                # transpose will have learnable param while nn.upsample does not
                nn.ConvTranspose2d(feature*2, feature, kernel_size=2, stride=2)  # *2 accouting for skip connection
            )

            self.ups.append(DoubleConv(feature*2, feature))

        # final conv 
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)


    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            # down part
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # bottleneck
        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        # up part
        for idx in range(0, len(self.ups), 2): # step of 2 to do up and double conv
            x = self.ups[idx](x)  #transpose
            skip_connection = skip_connections[idx//2]  #2 to counterfit the step of two 

            # takes care of the pooling floor issue, could crop or pad instead
            # x is always going to be potentially smaller than skip connection input
            if skip_connection.shape != x.shape:
                x= TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)  # double conv
        
        return self.final_conv(x)
    

# testing if input == output shape   
def test():
    x = torch.rand((16, 3, 160, 160))  # b x c x w x h
    model = UNET(in_channels=3, out_channels=1)
    pred = model(x)
    print(x.shape)
    print(pred.shape)

if __name__ == "__main__":
    test()
