# -*- coding: utf-8 -*-

import numpy as np
import sklearn.preprocessing
import sklearn.model_selection._split
import sklearn.svm
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.colors

# Data load
data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)/Social_Network_Ads.csv')
x, y = data.iloc[:, 2:-1].values, data.iloc[:, -1].values

# Split train - test
x_train, x_test, y_train, y_test = sklearn.model_selection._split.train_test_split(x, y, train_size = 0.7)

# Feature scaling
scaler = sklearn.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# SVM Classification
svm = sklearn.svm.SVC()
svm.fit(x_train, y_train)
y_pred = svm.predict(x_test)

# Make grid for plotting
def make_grid(v, steps = 100):
    v_min_limit = v.min() - 0.5
    v_max_limit = v.max() + 0.5
    step_len = (v_max_limit - v_min_limit) / steps
    return np.arange(v_min_limit, v_max_limit, step_len)

x_0_grid = make_grid(x_test[:, 0])
x_1_grid = make_grid(x_test[:, 1])
x_0_mesh, x_1_mesh = np.meshgrid(x_0_grid, x_1_grid)
x_0_1_grid = np.array([x_0_mesh.ravel(), x_1_mesh.ravel()]).T
y_grid = svm.predict(x_0_1_grid)

# Unscale features
y_grid = y_grid.reshape(x_0_mesh.shape)

# Display plot
plt.contourf(x_0_mesh, x_1_mesh, y_grid, cmap = matplotlib.colors.ListedColormap(('red', 'green')), alpha = 0.2)
plt.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], color = 'green')
plt.scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1], color = 'red')
plt.show()