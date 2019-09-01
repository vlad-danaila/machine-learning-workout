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
    mean_diff = mean_diff.reshape(4, 1) # Make a column vector
    diff = x_cls - mean_cls # N x 4 - 4 = N x 4
    within_class_scatter += diff.T.dot(diff) # 4 x N dot N x 4 = 4 x 4
    between_class_scatter += len(x_cls) * mean_diff.dot(mean_diff.T)
    
    