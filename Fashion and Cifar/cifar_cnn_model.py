import torch.nn.functional as F
import torch.nn as nn

# this time we omit sequential to showcase a user defined module
class cnn_model_cifar(nn.Module):
    def __init__(self, num_classes):
        super(cnn_model_cifar, self).__init__()

        # conv layers
        # i_out = i_in + 2p -2
        # https://docs.huihoo.com/theano/0.9/tutorial/conv_arithmetic.html
        # user summary(mode, input_shape(color, width, height)) 
        # from torchsummary import summary to get the output size automatically
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2)) # 16 (32 x 32 is input size)

        self.conv2 =  nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2)) # 8

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, kernel_size=3,padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(2)) # 4

        # dense layers
        self.fc1 = nn.Linear(128*4*4, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 128*4*4)
        x = F.dropout(x, p=0.3)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=0.2)
        x = self.fc2(x)

        return x