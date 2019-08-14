# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.preprocessing as prepross
import sklearn.model_selection._split as split

dataset = pd.read_csv('C:\DOC\Workspace\Machine Learning A-Z Template Folder\Part 1 - Data Preprocessing\Data.csv')

x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Handle missing data
imputer = prepross.Imputer()
x[:, 1:3] = imputer.fit_transform(x[:, 1:3])

# Encoding categorical data
label_encoder = prepross.LabelEncoder()
x[:, 0] = label_encoder.fit_transform(x[:, 0])
y = label_encoder.fit_transform(y)

# One hot 
one_hot_encoder_x = prepross.OneHotEncoder(categorical_features = [0])
x = one_hot_encoder_x.fit_transform(x).toarray()

# Remove first column(categorical data trap)
x = np.delete(x, 0, axis = 1)

# Split test - train
x_train, x_test, y_train, y_test = split.train_test_split(x, y, test_size = 0.3)

# Scaling
scaler = prepross.StandardScaler()
x_train[:, 2:] = scaler.fit_transform(x_train[:, 2:])
x_test[:, 2:] = scaler.transform(x_test[:, 2:])

