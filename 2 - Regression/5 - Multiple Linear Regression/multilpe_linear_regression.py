# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import sklearn.model_selection._split
import sklearn.preprocessing as preprocess
import sklearn.linear_model

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')
x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

# Categorical data
label_encoder = preprocess.LabelEncoder()
one_hot = preprocess.OneHotEncoder(categorical_features = [3])
x[:, 3] = label_encoder.fit_transform(x[:, 3])
x = one_hot.fit_transform(x).toarray()
# dummy variable trap
x = x[:, 1:]

# Test train split
x_train, x_test, y_train, y_test = sk.model_selection._split.train_test_split(x, y, train_size = 0.7)

# Scale data
scaler = preprocess.StandardScaler()
x_train[:, 2:] = scaler.fit_transform(x_train[:, 2:])    
x_test[:, 2:] = scaler.transform(x_test[:, 2:])

# Regression
regressor = sk.linear_model.LinearRegression()
regressor.fit(x_train, y_train)
pred_test = regressor.predict(x_test)

