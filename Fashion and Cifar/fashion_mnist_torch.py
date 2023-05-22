import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
import plotting
import os.path
from datetime import datetime
from sklearn.metrics import confusion_matrix

train_dataset = torchvision.datasets.FashionMNIST(
    root = './Fashion and Cifar',
    train=True,
    transform=transforms.ToTensor(),
    download=True
)

test_dataset = torchvision.datasets.FashionMNIST(
    root = './Fashion and Cifar',
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

# check the data
print("samples, height, width", train_dataset.data.shape)
classes = np.unique(train_dataset.targets)
num_class = len(set(train_dataset.targets.numpy()))
print(np.unique(train_dataset.targets))

# model definition usually in a seperate py file but since this is so short it goes here 
class CNN(nn.Module):
    def __init__(self, num_class):
        super(CNN, self).__init__()
        self.conv_layers = nn.Sequential(
            # 1 input layer due to grayscale
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, stride=2), # 13
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2), # 6
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2), # 2 output width same for height
            nn.ReLU(),
        )
        # i = img width (square input), k = kernel width (square kernel), s = stride, p = zero padding
        # floor[((i-k)/s)] +1
        # https://docs.huihoo.com/theano/0.9/tutorial/conv_arithmetic.html
        #
        self.dense_layers = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(128*2*2, 512),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(512, num_class)
        )

    def forward(self, X):
        out = self.conv_layers(X)
        out = out.view(out.size(0), -1) # or use a flatten layer after the last relu in conv_layers
        out = self.dense_layers(out)
        return out

model = CNN(num_class)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(device)
model.to(device)

batch_size = 256
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset, 
                                           batch_size=batch_size, 
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset, 
                                          batch_size=batch_size, 
                                          shuffle=False)

# training loop batch gradient descent
def train(model, loss_func, optimizer, train_loader, test_loader, epochs, save = True):
    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)
    train_accs = np.zeros(epochs)
    test_accs = np.zeros(epochs)

    for it in range(epochs):
        # drop out/batch norm and others works differently in train and eval so we have to set the modes
        model.train()   
        train_loss =[]
        test_loss = []
        train_acc = []
        test_acc = []

        t_0 = datetime.now()

        for inputs, targets in train_loader:
            # push data to gpu
            inputs = inputs.to(device)
            targets = targets.to(device)

            # set parameter gradients to zero 
            optimizer.zero_grad()

            # forward pass
            outputs = model(inputs)
            loss = loss_func(outputs, targets)

            # backward pass + optimize
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())

        train_loss = np.mean(train_loss) # avg loss per epoch over all batches
        train_losses[it] = train_loss

        model.eval()
        
        n_correct = 0.
        n_total = 0.

        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            output = model(inputs)

            # Get predictions
            # torch.max returns both max and argmax
            _, predictions = torch.max(output, 1)
            
            # update counts
            n_correct += (predictions == targets).sum().item()
            n_total += targets.shape[0]
            acc = n_correct / n_total
            test_acc.append(acc)

            loss = loss_func(output, targets)
            test_loss.append(loss.item())

        test_acc = np.mean(test_acc)
        test_accs[it] = test_acc
        test_loss = np.mean(test_loss)
        test_losses[it] = test_loss

        n_correct = 0.
        n_total = 0.
        # quick train loop in eval mode to get the accuracy 
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            output = model(inputs)

            # Get predictions
            # torch.max returns both max and argmax
            _, predictions = torch.max(output, 1)
            
            # update counts
            n_correct += (predictions == targets).sum().item()
            n_total += targets.shape[0]

        train_acc = n_correct / n_total
        train_acc = np.mean(train_acc)
        train_accs[it] = train_acc
       

        dt = datetime.now() - t_0
        print(f' Epoch {it+1}/{epochs}, Train Loss:{train_loss: .4f}, Test Loss:{test_loss: .4f} , Duration: {dt}')

    if save:
        torch.save(model.state_dict(), 'trained_model.pt')

    return train_losses, test_losses, train_accs, test_accs

# train only if there is no saved model
if not os.path.isfile('.\\Fashion and Cifar\\trained_model.pt'):
    train_losses, test_losses, train_accs, test_accs = train(model, loss_func, optimizer, train_loader, test_loader, epochs = 18, save=True)

    plotting.plot_train_test_loss(train_losses, test_losses)
    plotting.plot_train_test_acc(train_accs, test_accs)

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
plotting.plot_confusion_matrix(cm, list(range(10)))