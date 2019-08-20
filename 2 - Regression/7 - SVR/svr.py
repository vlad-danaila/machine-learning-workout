# -*- coding: utf-8 -*-
"""
Created on Tue Aug 20 17:26:46 2019

@author: VLADRARESDANAILA
"""

import numpy as np
import sklearn as sk
import pandas as pd
import sklearn.preprocessing
import sklearn.svm
import matplotlib.pyplot as plt

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Position_Salaries.csv')
x, y = data.iloc[:, 1].values, data.iloc[:, -1].values
x, y = np.expand_dims(x, 1), np.expand_dims(y, 1)

# Feature scaling
x_scaler, y_scaler = sk.preprocessing.StandardScaler(), sk.preprocessing.StandardScaler()
x, y = x_scaler.fit_transform(x), y_scaler.fit_transform(y)

# SVR
regressor = sk.svm.SVR()
regressor.fit(x, y.ravel())

# Prediction
career_level = np.arange(1, 10, 0.001)
career_level = np.expand_dims(career_level, 1)
career_level_scaled = x_scaler.transform(career_level)
predictions = regressor.predict(career_level_scaled)
predictions = y_scaler.inverse_transform(predictions)

# Plot
initial_x = x_scaler.inverse_transform(x).ravel()
initial_y = y_scaler.inverse_transform(y).ravel()
plt.scatter(initial_x, initial_y, color = 'red')
plt.plot(career_level, predictions)
plt.title('SVR in action')
plt.xlabel('Career')
plt.ylabel('Salary')
plt.show()