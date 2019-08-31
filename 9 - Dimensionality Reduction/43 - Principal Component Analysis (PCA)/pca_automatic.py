# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import sklearn.preprocessing
import sklearn.decomposition
import sklearn.model_selection._split
import sklearn.linear_model
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 9 - Dimensionality Reduction/Section 43 - Principal Component Analysis (PCA)/Wine.csv')
x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

# Split
x_train, x_test, y_train, y_test = sk.model_selection._split.train_test_split(x, y, train_size = 0.8)

# Scale
scaler = sk.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# PCA
pca = sk.decomposition.PCA(n_components = 2)
x_train = pca.fit_transform(x_train)
x_test = pca.transform(x_test)

# Fit linear classifier
classifier = sk.linear_model.LogisticRegression()
classifier.fit(x_train, y_train)

# Predict
y_pred = classifier.predict(x_test)