# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.preprocessing
import matplotlib.pyplot as plt

# Loading data
iris_dataset = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
data = pd.io.parsers.read_csv(filepath_or_buffer = iris_dataset, header = None)
data.dropna(how = "all", inplace = True) 
x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

# Encode classes for y
label_encoder = sk.preprocessing.LabelEncoder()
y = label_encoder.fit_transform(y)

# Separate x per classes
x_class = [x[y == i] for i in range(3)]
    
# Init scatter matrices
within_class_scatter = np.zeros((4,4))
between_class_scatter = np.zeros((4, 4))
    
# Mean of features for all classes
mean_all = x.mean(axis = 0) # 4
for x_cls in x_class:
    # Mean of features per class
    mean_cls = np.mean(x_cls, axis=0) # 4 
    mean_diff = mean_cls - mean_all # 4 - 4 = 4
    mean_diff = mean_diff.reshape(4, 1) # Make a column vector 4 x 1
    diff = x_cls - mean_cls # N x 4 - 4 = N x 4
    within_class_scatter += diff.T.dot(diff) # 4 x N dot N x 4 = 4 x 4
    between_class_scatter += len(x_cls) * mean_diff.dot(mean_diff.T) # 4 x 1 dot 1 x 4 = 4 x 4 
    
# Create final matrix and perform eigen decomposition
# We want to minimize within class variance and maximize between class variance
scatter_div = np.linalg.inv(within_class_scatter).dot(between_class_scatter) # 4 x 4
eigen_vals, eigen_vect = np.linalg.eig(scatter_div) # 4, 4 x 4

# Get only the first 2 components
lda_matrix = eigen_vect[:, :2]

# Transform x
x_transformed = x.dot(lda_matrix)

# Plot
colors = 'red', 'green', 'blue'
for i in range(3):
    plt.scatter(x_transformed[y == i][:, 0], x_transformed[y == i][:, 1], color = colors[i])
plt.title('LDA')
plt.show()