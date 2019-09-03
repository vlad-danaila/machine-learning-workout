# -*- coding: utf-8 -*-

import xgboost
import numpy as np
import sklearn as sk
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import matplotlib.pyplot
import matplotlib.colors

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 10 - Model Selection & Boosting/Section 49 - XGBoost/Churn_Modelling.csv')
x, y = data.iloc[:, 3:-1].values, data.iloc[:, -1].values

# Train test split
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, train_size = 0.8)

# Scaling
scaled_columns = [0, 3, 4, 5, 6, 9]
scaler = sk.preprocessing.StandardScaler()
scaler.fit(x_train[:, scaled_columns])

# Categorical data
one_hot_columns = [1, 2]
one_hot_encoders = [sk.preprocessing.OneHotEncoder(sparse = False) for i in one_hot_columns] 
for i in range(len(one_hot_columns)):
    one_hot_encoders[i].fit(x_train[:, [one_hot_columns[i]]])

# Data preparation
def prepare_data(x):
    x[:, scaled_columns] = scaler.transform(x[:, scaled_columns])
    for i in range(len(one_hot_columns)):
        one_hot = one_hot_encoders[i].transform(x[:, [one_hot_columns[i]]])
        one_hot = np.delete(one_hot, 0, axis = 1)
        x = np.hstack((x, one_hot))
    x = np.delete(x, one_hot_columns, axis = 1)
    x = x.astype(np.float64)
    return x

x_train, x_test = prepare_data(x_train), prepare_data(x_test)

# Define classifier
model = xgboost.XGBClassifier()

# Asses accuracy
f1_scores = sk.model_selection.cross_val_score(model, x_train, y_train, scoring = 'f1', cv = 10, n_jobs = -1)
print('F1 mean is', f1_scores.mean())
print('F1 standard variation is', f1_scores.std())