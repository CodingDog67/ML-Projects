"""
Sample case of a simple CNN in functional API style - Tensorflow/Keras, confusion matrix code can be reused nicely
"""


import os.path
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import tensorflow as tf
print (tf.__version__)
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.python.client import device_lib
device_lib.list_local_devices()

print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, Dropout
from tensorflow.keras.models import Model 
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix
import itertools

# functional API example
# easier to create branches / models with multiple input/outputs

# loading the data
fashion_mnist = tf.keras.datasets.fashion_mnist
print(type(fashion_mnist))
(x_train, y_train),(x_test, y_test) = fashion_mnist.load_data()
print("sample number, height, width: ", x_train.shape)

# rearrange to fit conv layer of height, width, color
x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

num_classes = len(set(y_train))
print('num of classes: ', num_classes)

# model specs
optim = 'adam'
loss = 'sparse_categorical_crossentropy'
metric = 'accuracy'

# building the model
i = Input(shape=x_train[0].shape) # single sample shape
x = Conv2D(32, (3,3), strides=2, activation='relu')(i) # num of filters, filtersize, stride, activation
x = Conv2D(64, (3,3), strides=2, activation='relu')(x) 
x = Conv2D(128, (3,3), strides=2, activation='relu')(x) 
x = Flatten()(x)
x = Dropout(0.2)(x)
x = Dense(512, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(num_classes, activation='softmax')(x)

if not os.path.isfile('D:\\Code\\ML Projects\\Fashion and Cifar\\fashion_model.h5'):
    model  = Model(i, x) # list of inputs/single input, list of outputs/single output
    model.compile(optimizer=optim, loss=loss, metrics=[metric])
    model_trained = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=20)
    scores = model.evaluate(x_test, y_test, verbose=0)

    # save model
    model.save('fashion_model.h5')
    print('Model Saved!')

    # plotting the results
    plt.plot(model_trained.history['accuracy'], label='acc')
    plt.plot(model_trained.history['val_accuracy'], label='val_acc')
    plt.title('Accuracy plot')
    plt.legend()
    plt.show()

    plt.plot(model_trained.history['loss'], label='acc')
    plt.plot(model_trained.history['val_loss'], label='val_acc')
    plt.title('Loss plot')
    plt.legend()
    plt.show()

else:
    # load model
    model_trained=load_model('D:\\Code\\ML Projects\\Fashion and Cifar\\fashion_model.h5')
    model_trained.summary()
    model_trained.evaluate(x_test, y_test)



def plot_confusion_matrix(cm, classes, normalize=False,  cmap=plt.cm.Blues):
    
    if normalize:
        print('Normalized confusion matrix')
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        print('Confusion Matrix not normalizs')
    
    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks,classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i,j], fmt), horizontalalignment='center',color='white' if cm[i,j] > thresh else 'black')

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

p_test = model_trained.predict(x_test).argmax(axis=1)
cm = confusion_matrix(y_test, p_test)
plot_confusion_matrix(cm, list(range(10)))
