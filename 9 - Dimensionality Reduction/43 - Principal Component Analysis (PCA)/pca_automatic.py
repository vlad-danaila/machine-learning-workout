# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.model_selection._split
import sklearn.linear_model
import sklearn.metrics
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 9 - Dimensionality Reduction/Section 43 - Principal Component Analysis (PCA)/Wine.csv')
x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

# Split
x_train, x_test, y_train, y_test = sk.model_selection._split.train_test_split(x, y, train_size = 0.8)

# Scale
scaler = sk.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# PCA
pca = sk.decomposition.PCA(n_components = 2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# Fit linear classifier
classifier = sk.linear_model.LogisticRegression()
classifier.fit(x_train, y_train)

# Predict
y_pred = classifier.predict(x_test)

# Confusion matrix
cm = sk.metrics.confusion_matrix(y_test, y_pred)
accuracy = np.sum(np.diag(cm)) / np.sum(cm)

# Plotting
def make_grid(x, steps = 1000):
    x_min, x_max = x.min(), x.max()
    diff = x_max - x_min
    padd = diff / 4
    return np.linspace(x_min - padd, x_max + padd, steps)

grid_x0, grid_x1 = make_grid(x_test[:, 0]), make_grid(x_test[:, 1])
grid_mesh_x0, grid_mesh_x1 = np.meshgrid(grid_x0, grid_x1)
grid_x0_x1 = np.array([grid_mesh_x0.ravel(), grid_mesh_x1.ravel()]).T
grid_y = classifier.predict(grid_x0_x1).reshape(grid_mesh_x0.shape)

colors = 'red', 'green', 'blue'

# Test points
for i in range(1, 4):
    is_class_i = y_test == i
    plt.scatter(x_test[is_class_i, 0], x_test[is_class_i, 1], color = colors[i - 1])
    
# Decision boundries    
color_map = matplotlib.colors.ListedColormap(colors)    
plt.contourf(grid_x0, grid_x1, grid_y, cmap = color_map, alpha = 0.2)    

plt.title('PCA')    
plt.xlabel('PCA projection 1')
plt.ylabel('PCA projection 2')

plt.show()