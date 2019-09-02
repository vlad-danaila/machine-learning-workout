# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.decomposition
import sklearn.linear_model
import sklearn.metrics
import matplotlib.pyplot as plt
import matplotlib.colors

# Load data
data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 9 - Dimensionality Reduction/Section 45 - Kernel PCA/Social_Network_Ads.csv')
x, y = data.iloc[:, 1:-1].values, data.iloc[:, -1].values

# Split data
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, train_size = 0.8)

# Gender label encoding
label_encoder = sk.preprocessing.LabelEncoder()
x_train[:, 0] = label_encoder.fit_transform(x_train[:, 0])
x_test[:, 0] = label_encoder.transform(x_test[:, 0])

# Scaling
scaler = sk.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Apply kernel pca
kpca = sk.decomposition.KernelPCA(n_components = 2, kernel = 'rbf')
x_train = kpca.fit_transform(x_train)
x_test = kpca.transform(x_test)

# Create linear classifier
classifier = sk.linear_model.LogisticRegression()
classifier.fit(x_train, y_train)

# Prediction
y_pred = classifier.predict(x_test)
cm = sk.metrics.confusion_matrix(y_test, y_pred)
accuracy = np.sum(np.diag(cm)) / np.sum(cm)

# Plot
colors = 'red', 'green'

def make_grid(x, steps = 100):
    x_min, x_max = x.min(), x.max()
    diff = x_max - x_min
    padd = diff / 5
    return np.linspace(x_min - padd, x_max + padd, steps)

grid_x0, grid_x1 = make_grid(x_test[:, 0]), make_grid(x_test[:, 1])
grid_mesh_x0, grid_mesh_x1 = np.meshgrid(grid_x0, grid_x1)
grid_x0_x1 = np.array([grid_mesh_x0.ravel(), grid_mesh_x1.ravel()]).T
grid_y = classifier.predict(grid_x0_x1).reshape(grid_mesh_x0.shape)

# Plot decision boundaries
color_map = matplotlib.colors.ListedColormap(colors)
plt.contourf(grid_x0, grid_x1, grid_y, alpha = 0.1, cmap = color_map)

# Plot true points scatter
for i in range(2):
    is_class_i = y_test == i
    plt.scatter(x_test[is_class_i, 0], x_test[is_class_i, 1], color = colors[i])
    
plt.show()