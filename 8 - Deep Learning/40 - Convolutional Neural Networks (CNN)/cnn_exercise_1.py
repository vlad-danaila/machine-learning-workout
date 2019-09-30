# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection
import matplotlib.pyplot as plt
import keras as k
import keras.preprocessing.image

DATASET_PATH = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/'
IMG_SIZE = 64, 64
BATCH_SIZE = 64

datagen_train = k.preprocessing.image.ImageDataGenerator(
        rotation_range = 20, 
        brightness_range = (.8, 1.2),
        shear_range = 0.2,
        zoom_range = 0.2,
        horizontal_flip = True,
        rescale = 1 / 255
)
datagen_test = k.preprocessing.image.ImageDataGenerator( rescale = 1 / 255 )

data_train = datagen_train.flow_from_directory(
    DATASET_PATH + 'training_set',
    IMG_SIZE,
    class_mode = 'binary',
    batch_size = BATCH_SIZE
)


