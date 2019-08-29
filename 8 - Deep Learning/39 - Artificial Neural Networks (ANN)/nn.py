# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection._split
import sklearn.metrics
import keras as ks
import keras.models
import keras.layers
import pandas as pd

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 8 - Deep Learning/Section 39 - Artificial Neural Networks (ANN)/Churn_Modelling.csv')

x, y = data.iloc[:, 3:-1].values, data.iloc[:, -1].values

x_train, x_test, y_train, y_test = sk.model_selection._split.train_test_split(x, y, train_size = .8)

# Prepare scaler
scaled_columns = [0, 3, 4, 5, 6, 9]
scaler = sk.preprocessing.StandardScaler()
scaler.fit(x[:, scaled_columns])

# Prepare one hot encoder
one_hot_columns = [1, 2]
non_one_hot_columns = list(set(range(len(x[0]))) - set(one_hot_columns))
one_hot_encoder = sk.preprocessing.OneHotEncoder(sparse = False)
one_hot_encoder.fit(x_train[:, one_hot_columns])

def process_data(x):
    x[:, scaled_columns] = scaler.transform(x[:, scaled_columns])
    one_hot_data = one_hot_encoder.transform(x[:, one_hot_columns])
    x = np.hstack((x[:, non_one_hot_columns], one_hot_data))
    return x

x_train = process_data(x_train)
x_test = process_data(x_test)
