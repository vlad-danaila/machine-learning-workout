# -*- coding: utf-8 -*-
"""
Created on Mon Aug 19 17:45:36 2019

@author: VLADRARESDANAILA
"""

import numpy as np
import sklearn as sk
import sklearn.preprocessing
import matplotlib.pyplot as plt
import sklearn.model_selection._split
import pandas as pd
import sklearn.linear_model

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv')
x = data.iloc[:, 1].values
y = data.iloc[:, 2].values

x = np.expand_dims(x, axis = 1)
y = np.expand_dims(y, axis = 1)

# Polinomial features
ploy = sk.preprocessing.PolynomialFeatures(degree = 5)
x_poly = ploy.fit_transform(x)

# Linear regression
regressor = sk.linear_model.LinearRegression()
regressor.fit(x_poly, y)
predicitons = regressor.predict(x_poly)

# Plots
x_plot = np.arange(x[0], x[-1], step = 0.0001)
x_plot = np.expand_dims(x_plot, 1)
predictions_plot = regressor.predict(ploy.fit_transform(x_plot))

plt.plot(x_plot, predictions_plot)
plt.scatter(x, y, color = 'red')
plt.show()