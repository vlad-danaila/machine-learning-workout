# -*- coding: utf-8 -*

import numpy as np
import sklearn.preprocessing
import sklearn.model_selection._split
import sklearn.neighbors
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
import sklearn.metrics

# Data load
data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 15 - K-Nearest Neighbors (K-NN)/Social_Network_Ads.csv')
x, y = data.iloc[:, 2:-1].values, data.iloc[:, -1].values

# Train test split
x_train, x_test, y_train, y_test = sklearn.model_selection._split.train_test_split(x, y, train_size = 0.7)

# KNN clasificatioin
knn = sklearn.neighbors.KNeighborsClassifier(n_neighbors = 5, p = 2)
knn.fit(x_train, y_train)

# Predictions test-set
y_pred = knn.predict(x_test)

# Confusion matrix & accuracy
cm = sklearn.metrics.confusion_matrix(y_test, y_pred)
accuracy = np.sum(np.diag(cm)) / np.sum(cm)
print('Accuracy is', accuracy)

# Making grid for plot
def get_grid(x, n_steps = 1000):
    x_min_limit, x_max_limit = x.min() - 1, x.max() + 1
    x_step = (x_max_limit - x_min_limit) / n_steps
    x_grid = np.arange(x_min_limit, x_max_limit, x_step)
    return x_grid

x_0_grid = get_grid(x[:, 0])
x_1_grid = get_grid(x[:, 1])
x_0_mesh, x_1_mesh = np.meshgrid(x_0_grid, x_1_grid)
x_0_1_grid = np.array([x_0_mesh.ravel(), x_1_mesh.ravel()]).T
y_pred_grid = knn.predict(x_0_1_grid).reshape(x_0_mesh.shape)

# Display plot
plt.contourf(x_0_grid, x_1_grid, y_pred_grid, alpha = 0.2, cmap = matplotlib.colors.ListedColormap(('green', 'red')))
plt.scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1], color = 'green')
plt.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], color = 'red')
plt.show()