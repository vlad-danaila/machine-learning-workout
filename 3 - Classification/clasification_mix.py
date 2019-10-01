# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
import sklearn.decomposition
import pandas as pd

DATASET_PATH = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)/Social_Network_Ads.csv'
data = pd.read_csv(DATASET_PATH)

# Data preprocessing
x, y = data.iloc[:, 1:-1].values, data.iloc[:, -1].values
x[:, 0] = sk.preprocessing.LabelEncoder().fit_transform(x[:, 0])

x = x.astype(np.float32)

x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size = .25)

scaler = sk.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define models
models = {}

logistic = sk.linear_model.LogisticRegression()
models['logistic'] = logistic


# PCA is aplied here only for visualisation
pca = sk.decomposition.PCA(2)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

def segmented(x, steps = 1000):
    min, max = np.min(x), np.max(x)
    padd = (max - min) / 5
    return np.linspace(min - padd, max + padd, steps)

x0, x1 = segmented(x_train_pca[:, 0]), segmented(x_train_pca[:, 1])
x0_mesh, x1_mesh = np.meshgrid(x0, x1)
x0_1 = np.c_[x0_mesh.ravel(), x1_mesh.ravel()]
x_grid = pca.inverse_transform(x0_1)

logistic.fit(x_train, y_train)
y_pred = logistic.predict(x_test)
conf_matrix = sk.metrics.confusion_matrix(y_test, y_pred)
accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)
y_grid = logistic.predict(x_grid).reshape(x0_mesh.shape)

plt.contourf(x0, x1, y_grid)
plt.scatter(x_test_pca[y_test == 0, 0], x_test_pca[y_test == 0, 1])
plt.scatter(x_test_pca[y_test == 1, 0], x_test_pca[y_test == 1, 1])

'''
# Train & evaluate models
for model_name, model in models.items():
    model.fit(x_train)
    '''