# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 13:09:22 2019

@author: VLADRARESDANAILA
"""

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.tree
import sklearn.ensemble
import sklearn.preprocessing
import matplotlib.pyplot as plt

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 8 - Decision Tree Regression/Position_Salaries.csv')
x, y = data.iloc[:, 1].values, data.iloc[:, -1].values
x, y = np.expand_dims(x, axis = 1), np.expand_dims(y, axis = 1)

# Decision tree
regressor_tree = sk.tree.DecisionTreeRegressor() 
regressor_tree.fit(x, y)

# Random forest
regressor_forest = sk.ensemble.RandomForestRegressor()
regressor_forest.fit(x, y)

# Prediction
x_grid = np.arange(x.min(), x.max(), step = 0.0001)
x_grid = x_grid.reshape((x_grid.shape[0], 1))
pred_tree = regressor_tree.predict(x_grid)
pred_forst = regressor_forest.predict(x_grid)

# Plot
plt.scatter(x, y, color = 'red')
plt.plot(x_grid, pred_tree, color = 'yellow')
plt.plot(x_grid, pred_forst, color = 'orange')
plt.title('Decision tree regression')
plt.xlabel('Career level')
plt.ylabel('salary')
plt.show()