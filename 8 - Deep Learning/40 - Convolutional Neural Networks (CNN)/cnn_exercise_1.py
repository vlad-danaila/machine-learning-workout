# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection
import matplotlib.pyplot as plt
import keras as k
import keras.preprocessing.image

# Data load

DATASET_PATH = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/'
IMG_SIZE = 64, 64
BATCH_SIZE = 64

datagen_train = k.preprocessing.image.ImageDataGenerator(
        rotation_range = 30, 
        brightness_range = (.8, 1.2),
        shear_range = 0.3,
        zoom_range = 0.5,
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

data_test = datagen_test.flow_from_directory(
    DATASET_PATH + 'test_set',
    IMG_SIZE,
    class_mode = 'binary',
    batch_size = BATCH_SIZE
)

def plot_img(img, label, prediction):
    plt.imshow(img)
    plt.title('dog' if prediction else 'cat', color = 'green' if label == prediction else 'red')

imgs, labels = next(iter(data_train))
fig = plt.figure(figsize = (14, 7))

for i in range(10):
    fig.add_subplot(2, 5, i + 1)
    plot_img(imgs[i], labels[i], labels[i])
  
# Define model

model = k.models.Sequential((
        k.layers.Conv2D(32, 3, padding='same', activation='relu', input_shape=(64, 64, 3)),
        k.layers.MaxPool2D(),
        k.layers.Conv2D(64, 3, padding='same', activation='relu'),
        k.layers.MaxPool2D(),
        k.layers.Conv2D(32, 3, padding='same', activation='relu'),
        k.layers.MaxPool2D(),
        k.layers.Flatten(),
        k.layers.Dense(1)
))

model.compile(k.optimizers.Adam(), k.losses.binary_crossentropy, [k.metrics.binary_accuracy])

model.fit_generator(data_train, steps_per_epoch = 1000, epochs = 10, validation_data = data_test, validation_steps = 200)