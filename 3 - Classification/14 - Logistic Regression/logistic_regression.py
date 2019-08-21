# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection._split
import sklearn.linear_model

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

