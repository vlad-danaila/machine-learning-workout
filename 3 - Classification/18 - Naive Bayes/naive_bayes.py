# -*- coding: utf-8 -*-
import numpy as np
import sklearn as sk
import sklearn.naive_bayes
import sklearn.preprocessing
import sklearn.model_selection._split
import pandas as pd
import sklearn.metrics
import matplotlib.pyplot as plt
import matplotlib.colors

# Data load
data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)/Social_Network_Ads.csv')
x, y = data.iloc[:, 2:-1].values, data.iloc[:, -1].values

# Data split
x_train, x_test, y_train, y_test = sk.model_selection._split.train_test_split(x, y, train_size = 0.7)

# Feature scaling
scaler = sk.preprocessing.StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Predictions
classifier = sk.naive_bayes.GaussianNB()
classifier.fit(x_train_scaled, y_train)
y_pred = classifier.predict(x_test_scaled)

# Confusion matrix and accuracy
cm = sk.metrics.confusion_matrix(y_test, y_pred)
accuracy = np.sum(np.diag(cm)) / np.sum(cm)
print('Accuracy is', accuracy)

# Make grid for plotting
def make_grid(x, steps = 100):
    x_min, x_max = x.min(), x.max()
    step_len = (x_max - x_min) / steps
    pading = steps / 4 * step_len
    return np.arange(x_min - pading, x_max + pading, step_len)

grid_x_0, grid_x_1 = make_grid(x_test[:, 0]), make_grid(x_test[:, 1])
grid_x_0_scaled, grid_x_1_scaled = make_grid(x_test_scaled[:, 0]), make_grid(x_test_scaled[:, 1])
grid_mesh_x_0_scaled, grid_mesh_x_1_scaled = np.meshgrid(grid_x_0_scaled, grid_x_1_scaled)
grid_mesh_x_0_1_scaled = np.array([grid_mesh_x_0_scaled.ravel(), grid_mesh_x_1_scaled.ravel()]).T
grid_y = classifier.predict(grid_mesh_x_0_1_scaled).reshape(grid_mesh_x_0_scaled.shape)

# Display plot
color_map = matplotlib.colors.ListedColormap(('red', 'green'))
plt.contourf(grid_x_0, grid_x_1, grid_y, alpha = 0.2, cmap = color_map)
plt.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], color = 'green')
plt.scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1], color = 'red')
plt.show()