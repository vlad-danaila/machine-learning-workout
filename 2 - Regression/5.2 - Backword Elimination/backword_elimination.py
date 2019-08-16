# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import statsmodels.api as sm
import sklearn.preprocessing as preprocess
import sklearn.model_selection._split as split

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Categorical data
category_index = 3
labelEncoder = preprocess.LabelEncoder()
one_hot = preprocess.OneHotEncoder(categorical_features = [category_index])
x[:, category_index] = labelEncoder.fit_transform(x[:, category_index])
x = one_hot.fit_transform(x).toarray()

# Dummy variable trap
x = x[:, 1:]

# Train - test split
x_train, x_test, y_train, y_test = split.train_test_split(x, y, train_size = 0.7)

# Feature scaling
scaler = preprocess.StandardScaler()
x_train[:, 2:5] = scaler.fit_transform(x_train[:, 2:5])
x_test[:, 2:5] = scaler.transform(x_test[:, 2:5])

# Backword elimination - manual

# Step 1
ols = sm.OLS(endog = y_train, exog = x_train).fit()
ols.summary()

