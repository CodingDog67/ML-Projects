# example of inference learning on a pretrained vgg16 model 
from __future__ import print_function, division

import numpy as np
import matplotlib.pyplot as plt

from keras_preprocessing import image
from keras_preprocessing.image import ImageDataGenerator
from keras.layers import Input, Dense, Flatten, Lambda
from keras.models import Model
from keras.applications.vgg16 import VGG16, preprocess_input
from sklearn.metrics import confusion_matrix

from glob import glob

#IMAGE_SIZE
