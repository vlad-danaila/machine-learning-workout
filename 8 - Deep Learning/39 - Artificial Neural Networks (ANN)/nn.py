# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection._split
import sklearn.metrics
import keras as ks
import keras.models
import keras.layers
import pandas as pd

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Churn_Modelling.csv')

x, y = data.iloc[:, 3:-1].values, data.iloc[:, -1].values

x_train, x_test, y_train, y_test = sk.model_selection._split.train_test_split(x, y, train_size = .8, shuffle = False)

# Prepare scaler
scaled_columns = [0, 3, 4, 5, 6, 9]
scaler = sk.preprocessing.StandardScaler()
scaler.fit(x[:, scaled_columns])

# Prepare one hot encoder
one_hot_columns = [1, 2]
non_one_hot_columns = list(set(range(len(x[0]))) - set(one_hot_columns))
one_hot_encoder = sk.preprocessing.OneHotEncoder(sparse = False)
one_hot_encoder.fit(x_train[:, one_hot_columns])

def process_data(x):
    x[:, scaled_columns] = scaler.transform(x[:, scaled_columns])
    one_hot_data = one_hot_encoder.transform(x[:, one_hot_columns])
    x = np.hstack((x[:, non_one_hot_columns], one_hot_data))
    return x

x_train, x_test = process_data(x_train), process_data(x_test)

# Create neural network
model = ks.models.Sequential((
    ks.layers.Dense(13, activation = 'relu', input_shape = (13,)),
    ks.layers.Dense(6, activation = 'relu'),  
    ks.layers.Dense(3, activation = 'relu'),
    ks.layers.Dense(1, activation = 'sigmoid'),        
))

# Train model
model.compile('adam', loss = ks.losses.binary_crossentropy, metrics = ['accuracy'])
model.fit(x_train, y_train, batch_size = 32, epochs = 10, validation_split = 0.3)

# Make predictions
y_pred = model.predict(x_test)
y_pred = y_pred > 0.5

confusion_matrix = sk.metrics.confusion_matrix(y_test, y_pred)
print('Confusion matrix', confusion_matrix, sep = '\n')

accuracy = np.sum(np.diag(confusion_matrix)) / np.sum(confusion_matrix)
print('Accuracy', accuracy, sep = '\n')