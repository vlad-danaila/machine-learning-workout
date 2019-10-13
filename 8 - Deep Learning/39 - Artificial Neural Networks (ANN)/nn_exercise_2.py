# -*- coding: utf-8 -*-
import keras as ks
import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import matplotlib.pyplot as plt

DATA_PATH = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Churn_Modelling.csv'
data = pd.read_csv(DATA_PATH)

# Preprocessing
geography = data['Geography']
gender = data['Gender']

one_hot = sk.preprocessing.OneHotEncoder(sparse = False, drop = 'first')

geography = one_hot.fit_transform(geography[:, np.newaxis])
gender = one_hot.fit_transform(gender[:, np.newaxis])

x = np.hstack((data['CreditScore'][:, np.newaxis], data.values[:, 6:-1], geography, gender))
y = data.values[:, -1]
x, y = x.astype(np.float32), y.astype(np.float32)

x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y)

scaler = sk.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define model
model = ks.Sequential([
    ks.layers.Dense(11, activation = ks.activations.relu, input_shape=(11,)),
    ks.layers.Dense(4, activation = ks.activations.relu),
    ks.layers.Dense(1),
])

optimizer = ks.optimizers.Adam()
loss = ks.losses.BinaryCrossentropy()
metrics = [ks.metrics.BinaryAccuracy(), ks.metrics.Precision(), ks.metrics.Recall()]

model.compile(optimizer, loss, metrics)

model.fit(x_train, y_train, batch_size = 32, epochs = 100, validation_data = (x_test, y_test))

y_pred = model.predict(x_test).reshape(-1) > .5
cm = sk.metrics.confusion_matrix(y_pred, y_test)
accuracy = np.sum(np.diag(cm)) / np.sum(cm)
print(accuracy)