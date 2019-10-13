# -*- coding: utf-8 -*-
import keras as ks
import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt

dataset_root_dir = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/'
dataset_classes_folders = ['cats', 'dogs']
img_size = 64, 64
batch_size = 64

data_gen_train = ks.preprocessing.image.ImageDataGenerator(
    rotation_range = 20,
    width_shift_range = .2,
    height_shift_range = .2,
    brightness_range = (.9, 1.1),
    shear_range = .1,
    zoom_range = (.8, 1.2),
    horizontal_flip = True,
    rescale = 1/255
)

generaotr_train = data_gen_train.flow_from_directory(
    directory = dataset_root_dir + '/training_set',
    target_size = img_size,
    classes = dataset_classes_folders,
    class_mode = 'binary',
    batch_size = batch_size,
    shuffle = True
)

# Testing the train generator
x, y = next(iter(generaotr_train))
plt.imshow(x[0])
plt.show()

data_gen_test = ks.preprocessing.image.ImageDataGenerator(
    rescale = 1/255
)

generaotr_test = data_gen_train.flow_from_directory(
    directory = dataset_root_dir + '/test_set',
    target_size = img_size,
    classes = dataset_classes_folders,
    class_mode = 'binary',
    batch_size = batch_size,
    shuffle = False
)

# Testing the test generator
x, y = next(iter(generaotr_test))
plt.imshow(x[0])
plt.show()

relu = ks.activations.relu

# Define model
model = ks.Sequential([
    ks.layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = relu, input_shape = (64, 64, 3)),
    ks.layers.MaxPool2D(), # size 32
        
    ks.layers.BatchNormalization(),    
    ks.layers.Dropout(.3),
    ks.layers.Conv2D(filters = 64, kernel_size = 3, padding = 'same', activation = relu), 
    ks.layers.MaxPool2D(), # size 16
    
    ks.layers.BatchNormalization(),    
    ks.layers.Dropout(.3),
    ks.layers.Conv2D(filters = 32, kernel_size = 3, padding = 'same', activation = relu), 
    ks.layers.MaxPool2D(), # size 8
        
    ks.layers.BatchNormalization(),    
    ks.layers.Dropout(.3),
    ks.layers.Conv2D(filters = 10, kernel_size = 3, padding = 'same', activation = relu), 
    ks.layers.MaxPool2D(), # size 4
    
    ks.layers.BatchNormalization(),    
    ks.layers.Flatten(),
    ks.layers.Dropout(.5),
    ks.layers.Dense(50),
    
    ks.layers.BatchNormalization(),
    ks.layers.Dense(1, activation = ks.activations.sigmoid)
])

optimizer = ks.optimizers.Adam()
loss = ks.losses.BinaryCrossentropy()
metrics = [ks.metrics.BinaryAccuracy()]

model.compile(optimizer, loss, metrics)

history = model.fit_generator(
        generaotr_train, 1000, 20, validation_data = generaotr_test, validation_steps = 32)


