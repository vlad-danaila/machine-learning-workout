# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection._split as split 
import sklearn.linear_model

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv')

x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

x_train, x_test, y_train, y_test = split.train_test_split(x, y, train_size = 0.7)

# Regression
regressor = sk.linear_model.LinearRegression()

regressor.fit(x_train, y_train)

pred_train = regressor.predict(x_train)
pred_test = regressor.predict(x_test)

# Plotting
plt.scatter(x_train, y_train)
plt.plot(x_train, pred_train, color = 'red')
plt.xlabel('years')
plt.ylabel('salary')
plt.title('Linerar regression')
plt.show()