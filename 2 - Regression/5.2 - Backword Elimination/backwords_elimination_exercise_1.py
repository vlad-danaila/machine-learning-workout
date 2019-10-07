# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.preprocessing
import statsmodels.api
import statsmodels

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')

# Data preprocessing
data_states = np.expand_dims(data['State'].values, axis = 1)
states = sk.preprocessing.OneHotEncoder(sparse = False, drop = 'first').fit_transform(data_states) 
x = np.hstack((data.values[:, 0:3], states, np.ones((50, 1))))
y = data.values[:, -1]
x, y = x.astype(np.float32), y.astype(np.float32)

columns = list(range(len(x[0])))
r = 0

for i in range(len(x[0])):
    print(columns)
    _x = x[:, columns]
    model = statsmodels.api.OLS(y, _x, 'raise')
    results = model.fit()
    #if results.rsquared_adj < r:
     #   break
    #r = results.rsquared_adj
    p_vals = np.array(results.pvalues)
    feat = np.argmax(p_vals)
    if p_vals[feat] <= .05:
        break
    columns = [c for c in columns if c != columns[feat]]
    
