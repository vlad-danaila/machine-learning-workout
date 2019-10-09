# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection
import xgboost

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 10 - Model Selection & Boosting/Section 49 - XGBoost/Churn_Modelling.csv')

# Data preporcessing

one_hot = sk.preprocessing.OneHotEncoder(drop = 'first', sparse = False)

geography = one_hot.fit_transform(data['Geography'].values[:, np.newaxis])
gender = one_hot.fit_transform(data['Gender'].values[:, np.newaxis])

x_columns = [3, 6, 7, 8, 9, 10, 11, 12]
x = np.hstack((data.values[:, x_columns], geography, gender)) 
x = x.astype(np.float32)

y = data.values[:, -1].astype(np.float32)

x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y)

scaler = sk.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define model
model = xgboost.XGBClassifier()
model.fit(x_train, y_train)

# Make predictions
y_pred = model.predict(x_test)
cm = sk.metrics.confusion_matrix(y_test, y_pred)
