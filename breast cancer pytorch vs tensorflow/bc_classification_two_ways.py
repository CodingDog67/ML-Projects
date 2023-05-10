import tensorflow as tf
print(tf.__version__)

import torch
print(torch.__version__)

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

import numpy as np

data = load_breast_cancer()
type(data)

D, input_shape = data.data.shape
# 1 take a look at the data, inspect the object
print('Number of datapoints / features: ', data.data.shape)
print('Classes: ', data.target_names)
print('Features: ', data.feature_names)

# 2 split the data
X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size = 0.33)

# 3 Scale/proprocess the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train) # fit because this is what is going to find the parameters (mean, std, etc)
X_test = scaler.transform(X_test) # use pre-fitted param to transform the data


# model Tensorflow version
model_tf = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(input_shape,)),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_tf.compile(optimizer = 'adam', loss='binary_crossentropy', metrics=['accuracy'])

tens_mod = model_tf.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

print('Train score: ', model_tf.evaluate(X_train, y_train) )
print('Test Score: ', model_tf.evaluate(X_test, y_test))


# model pytorch
model_torch = torch.nn.Linear(input_shape, 1)

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model_torch.parameters())

def fit_torch(model, criterion, optimizer, X_train, y_train, X_test, y_test, n_epochs=150):

    train_losses = np.zeros(n_epochs)
    test_losses = np.zeros(n_epochs)

    for it in range(n_epochs):
        output = model(X_train)
        loss = criterion(output, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output_test = model(X_test)
        loss_test = criterion(output_test, y_test)

        train_losses[it] = loss.item()
        test_losses[it] = loss_test.item()

        if (it+1)%10 == 0:
            print(f'Epoch: {it+1}/{n_epochs}, Training loss: {loss.item(): .4f}, Test loss: {loss_test.item(): .4f}')
    # evalutation    
    with torch.no_grad():
        p_train= model(X_train)
        p_train = (p_train.numpy() >0)
        
        p_test= model(X_test)
        p_test = (p_test.numpy() >0)

        train_acc = np.mean(y_train.numpy() == p_train)
        test_acc = np.mean(y_test.numpy() == p_test)

    return train_losses, test_losses, train_acc, test_acc

X_train = torch.from_numpy(X_train.astype(np.float32))
X_test = torch.from_numpy(X_test.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32)).reshape(-1, 1)
y_test = torch.from_numpy(y_test.astype(np.float32)).reshape(-1, 1)

train_losses, test_losses, train_acc, test_acc = fit_torch(model_torch, criterion, optimizer, X_train, y_train, X_test, y_test)

# plot the results
# Tensorflow
plt.plot(tens_mod.history['loss'], label='loss tensorflow')
plt.plot(tens_mod.history['val_loss'], label='val_loss  tensorflow')
plt.legend()
plt.show()

plt.plot(train_losses, label='loss pytorch')
plt.plot(test_losses, label='val_loss pytorch')
plt.legend()
plt.show()

plt.plot(tens_mod.history['accuracy'], label='acc')
plt.plot(tens_mod.history['val_accuracy'], label='val_acc')
plt.legend()
plt.show()

