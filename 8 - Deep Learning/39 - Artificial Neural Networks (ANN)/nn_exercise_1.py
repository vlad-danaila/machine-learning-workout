# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import keras as k
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.preprocessing.label
import sklearn.model_selection

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Churn_Modelling.csv')
x, y = data.iloc[:, 3:-1].values, data.iloc[:, -1].values 

# Data preprocessing

# Categorical columns
CATEGORICAL_COLUMNS = [1, 2]
one_hot_encoder = sk.preprocessing.OneHotEncoder()
categories = one_hot_encoder.fit_transform(x[:, CATEGORICAL_COLUMNS]).toarray()
categories = np.delete(categories, [0, 2], axis = 1)
x = np.delete(x, CATEGORICAL_COLUMNS, axis = 1)
x = x.astype(np.float32)
x = np.hstack((x, categories))

# Data split
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, train_size = 0.7)

# Normalization
SCALED_COLUMNS = [0, 1, 2, 3, 4, 7]
scaler = sk.preprocessing.StandardScaler()
x_train[:, SCALED_COLUMNS] = scaler.fit_transform(x_train[:, SCALED_COLUMNS])
x_test[:, SCALED_COLUMNS] = scaler.fit_transform(x_test[:, SCALED_COLUMNS])

# Define model
FEATURE_LEN = len(x_train[0])
model = k.models.Sequential([
    k.layers.Dense(16, activation=k.activations.relu,  input_shape=(FEATURE_LEN,)),        
    k.layers.Dense(8, activation=k.activations.relu),
    k.layers.Dense(1)
])
model.compile(k.optimizers.Adam(), k.losses.binary_crossentropy, metrics=[k.metrics.binary_accuracy])

