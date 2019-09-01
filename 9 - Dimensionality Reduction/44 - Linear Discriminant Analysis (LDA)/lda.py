# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection._split
import sklearn.linear_model
import sklearn.discriminant_analysis
import sklearn.metrics
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 9 - Dimensionality Reduction/Section 43 - Principal Component Analysis (PCA)/Wine.csv')
x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

x_train, x_test, y_train, y_test = sk.model_selection._split.train_test_split(x, y, train_size = 0.8)

scaler = sk.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

lda = sk.discriminant_analysis.LinearDiscriminantAnalysis(n_components = 2)
x_train = lda.fit_transform(x_train, y_train)
x_test = lda.transform(x_test)

classifier = sk.linear_model.LogisticRegression()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

cm = sk.metrics.confusion_matrix(y_test, y_pred)
accuracy = np.sum(np.diag(cm)) / np.sum(cm)

def make_grid(x, steps = 1000):
    x_min, x_max = x.min(), x.max()
    diff = x_max - x_min
    padd = diff / 5
    return np.linspace(x_min - padd, x_max + padd, steps)

grid_x0, grid_x1 = make_grid(x_test[:, 0]), make_grid(x_test[:, 1])
grid_mesh_x0, grid_mesh_x1 = np.meshgrid(grid_x0, grid_x1)
grid_x0_x1 = np.array([grid_mesh_x0.ravel(), grid_mesh_x1.ravel()]).T
grid_y = classifier.predict(grid_x0_x1).reshape(grid_mesh_x0.shape)

colors = 'red', 'green', 'blue'

for i in range(3):
    is_class = y_test == i + 1
    plt.scatter(x_test[is_class, 0], x_test[is_class, 1], color = colors[i])
    
color_map = matplotlib.colors.ListedColormap(colors)
plt.contourf(grid_x0, grid_x1, grid_y, cmap = color_map, alpha = 0.1)
    
plt.show()