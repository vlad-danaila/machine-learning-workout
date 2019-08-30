# -*- coding: utf-8 -*-

import numpy as np
import keras as ks
import keras.models
import keras.layers
import keras.preprocessing.image
import matplotlib.pyplot as plt

ROOT = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 40 - Convolutional Neural Networks (CNN)/dataset/'
IMAGE_DIMENSIONS = 64, 64
BATCH_SIZE = 32
CLASS_MODE = 'binary'

# Data preprcessing and loading
train_datagen = ks.preprocessing.image.ImageDataGenerator(
            rescale = 1./255,
            shear_range = 10,
            zoom_range = 0.2,
            horizontal_flip = True)
  
test_datagen = ks.preprocessing.image.ImageDataGenerator(rescale = 1./255)
  
train_generator = train_datagen.flow_from_directory(
          ROOT + 'training_set',
          target_size = IMAGE_DIMENSIONS,
          batch_size = BATCH_SIZE,
          class_mode = CLASS_MODE)
  
validation_generator = test_datagen.flow_from_directory(
          ROOT + 'test_set',
          target_size = IMAGE_DIMENSIONS,
          batch_size = BATCH_SIZE,
          class_mode = CLASS_MODE)
  
# Display an image as example
img_sample = next(train_generator)[0][0]
plt.imshow(img_sample)

# Define model
model = ks.models.Sequential((
        
        ks.layers.Conv2D(
                #input_shape=(64, 64, 3),
                filters = 32, 
                kernel_size = 3, 
                strides = 1,
                activation = 'relu',
                padding = 'same'
        ),
        
        ks.layers.MaxPool2D(),
        
        ks.layers.Conv2D(
                filters = 32, 
                kernel_size = 3, 
                strides = 1,
                activation = 'relu',
                padding = 'same'
        ),        
        
        ks.layers.MaxPool2D(),
        
        ks.layers.Flatten(),
        
        ks.layers.Dense(128, activation = ks.activations.relu),
        
        ks.layers.Dense(1, activation = ks.activations.sigmoid)

))

# Complie model
model.compile(
        optimizer = 'adam', 
        loss = ks.losses.binary_crossentropy, 
        metrics = ['accuracy'])

# Fit model
model.fit_generator(
          train_generator,
          steps_per_epoch = 8000,
          epochs = 10,
          validation_data = validation_generator,
          validation_steps = 2000)

# Make a prediction
img_samples = np.expand_dims(img_sample, axis = 0)
y_pred = model.predict(img_samples)
print('This is a', 'dog' if y_pred > .5 else 'cat')