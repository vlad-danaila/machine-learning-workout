# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
import sklearn.svm
import sklearn.decomposition

DATASET_PATH = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv'

# Data loading & preprocessing

# Categorical features
data = pd.read_csv(DATASET_PATH)
x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values
one_hot_encoder = sk.preprocessing.OneHotEncoder(sparse = False)
countries = one_hot_encoder.fit_transform(x[:, np.newaxis, -1])
x = np.hstack((x[:, :-1], countries))
x = x.astype(np.float64)

# Train - test split
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size = 1/5)

# Scaling
scaler = sk.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define models
models = {}
models['linear'] = sk.linear_model.LinearRegression()
models['decision tree'] = sk.tree.DecisionTreeRegressor()
models['random forest'] = sk.ensemble.RandomForestRegressor()
models['SVR'] = sk.svm.SVR()

model = sk.linear_model.LinearRegression()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# PCA only for visualization
pca = sk.decomposition.PCA(1)
x_train_pca = pca.fit_transform(x_train)
x_test_pca = pca.transform(x_test)

plt.scatter(x_train_pca, y_train)
