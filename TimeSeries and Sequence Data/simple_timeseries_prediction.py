# Simple sample of an autoregressive model compared to a RNN, focus is on data processsing and making the forcast correctly on a synthetic dataset. Note that sklearn has LinearRegression/ForecasterAutoreg, but this sample is solely in pytorch 

import torch 
import torch.nn as nn 
import numpy as np
import matplotlib.pyplot as plt

# 0 for Autogeression and 1 for RNN 
choose_model = 0 

# Created the dataset
n = 1000

#result without noise will perfect with the correct method, without we can still predict the periodically nature
series = np.sin(0.05*np.arange(n)) + np.random.randn(n)*0.03  # normal sin wave plus noise, 


# Sanatity check by plotting
plt.plot(series)
plt.show()

# building the dataset
T = 10 # amount of past values to predict the next value
X = []
Y = []

# split the data into time-windows of 10 stride 1
for t in range(len(series)-T):
    x = series[t:t+T]
    X.append(x)
    y = series[t+T] # 11th value as label
    Y.append(y)

X = np.array(X).reshape(-1, T)
Y = np.array(Y).reshape(-1, 1)
print("X Shape and Y Shape", X.shape, Y.shape)

N = len(X)

# Autoregressive model
model = nn.Linear(T, 1)
loss_func = nn.MSELoss()

# RNN model 


optim  = torch.optim.Adam(model.parameters(), lr = 0.05)


# training and test data split 50/50
x_train = torch.from_numpy(X[:-(N//2)].astype(np.float32)) 
y_train = torch.from_numpy(Y[:-(N//2)].astype(np.float32))
x_test = torch.from_numpy(X[-(N//2):].astype(np.float32))
y_test = torch.from_numpy(Y[-(N//2):].astype(np.float32))

# train on the full dataset due to small size
def train(model, loss_func, optim, x_train, y_train, x_test,y_test, epochs=200):

    train_losses = np.zeros(epochs)
    test_losses = np.zeros(epochs)

    for it in range(epochs):
        optim.zero_grad()

        outputs = model(x_train)
        loss = loss_func(outputs, y_train)

        loss.backward()
        optim.step()

        train_losses[it] = loss.item()

        # test loss 
        output_test = model(x_test)
        test_loss = loss_func(output_test, y_test)
        test_losses[it] = test_loss.item()

        print(f"Epoch {it+1}/{epochs}, Train loss: {loss.item():.4f}, Test loss: {test_loss.item():.4f}")

    return train_losses, test_losses

train_losses, test_losses = train(model, loss_func, optim, x_train, y_train, x_test,y_test, epochs=200)

# plot loss
plt.plot(train_losses, label='train_loss')
plt.plot(test_losses, label='test loss')
plt.legend()
plt.show()

# predictions 
validation_predictions = []

# First entry of x_test  
updated_x = torch.from_numpy(X[-(N//2)].astype(np.float32)) 


# POPULAR MISTAKE : input_ = X_test[i].view(1, -1); i+=1; basically forcasting on the true targets

while len(validation_predictions) < len(y_test.tolist()):

    _input = updated_x.view(1, -1) # reshape into 2D array, 1(sample) X T (features)
    p = model(_input) # index because model returns N x K (output node) (which here is 1x1), item to bring it back to python from tensor

    # update the predictions
    validation_predictions.append(p[0,0].item())

    # update input with latest forcast prediction
    updated_x = torch.cat((updated_x[1:], p[0]))

plt.plot(y_test.tolist(), label="true label")
plt.plot(validation_predictions, label="prediction")
plt.legend()
plt.show()