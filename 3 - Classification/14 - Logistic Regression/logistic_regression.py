# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection._split
import sklearn.linear_model
import matplotlib.pyplot as plt
import matplotlib.colors

# Load data
data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 14 - Logistic Regression/Social_Network_Ads.csv')
x, y = data.iloc[:, [2, 3]].values, data.iloc[:, -1].values

# Data split
x_train, x_test, y_train, y_test = sk.model_selection._split.train_test_split(x, y, train_size = 0.7)

# Feature scaling
scaler = sk.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Logistic regression
classifier = sklearn.linear_model.LogisticRegression()
classifier.fit(x_train, y_train)

# Making predictions
pred = classifier.predict(x_test)

# Confusion matrix
cm = sk.metrics.confusion_matrix(y_test, pred)
accuracy = np.sum(np.diag(cm)) / np.sum(cm)
print('Accuracy is', accuracy)

# Plot
x_0_coord = np.arange(x_train[:, 0].min(), x_train[:, 0].max(), 0.01)
x_1_coord = np.arange(x_train[:, 1].min(), x_train[:, 1].max(), 0.01)
grid_x_0, grid_x_1 = np.meshgrid(x_0_coord, x_1_coord)
grid_x_0_1 = np.array([grid_x_0.ravel(), grid_x_1.ravel()]).T
grid_pred = classifier.predict(grid_x_0_1).reshape(grid_x_0.shape)
plt.contourf(grid_x_0, grid_x_1, grid_pred, cmap = matplotlib.colors.ListedColormap(('red', 'green')))
plt.scatter(x_test[y_test == 0, 0], x_test[y_test == 0, 1], color = 'darkred')
plt.scatter(x_test[y_test == 1, 0], x_test[y_test == 1, 1], color = 'darkgreen')
plt.show()

