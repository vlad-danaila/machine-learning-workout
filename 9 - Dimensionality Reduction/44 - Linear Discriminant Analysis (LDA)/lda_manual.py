# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.preprocessing

# Loading data
iris_dataset = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
data = pd.io.parsers.read_csv(filepath_or_buffer = iris_dataset, header = None)
data.dropna(how = "all", inplace = True) 
x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

# Encode classes for y
label_encoder = sk.preprocessing.LabelEncoder()
y = label_encoder.fit_transform(y)

# Means 
mean_vectors = [np.mean(x[y == i], axis=0) for i in range(3)] # 3 x 4 matrix(class x features)
mean_features = x.mean(axis = 0) # 4
mean_diff = mean_vectors - mean_features # 3 x 4 minus 4

# Init scatter matrices
within_class_scatter = np.zeros((4,4))
between_class_scatter = np.zeros((4, 4))

for i in range(3):  
    # Compute within class scatter matrix
    x_class_i = x[y == i]
    diff = x_class_i - mean_vectors[i] # N x 4 minus 4, broadcasts difference to N
    within_class_scatter += diff.T.dot(diff) # 4 x N dot N x 4 gives 4 x 4 
    # Compute between class scatter matrix
    between_class_scatter += len(x_class_i) * mean_diff[i].reshape(4, 1).dot(mean_diff[i].reshape(1, 4))    
