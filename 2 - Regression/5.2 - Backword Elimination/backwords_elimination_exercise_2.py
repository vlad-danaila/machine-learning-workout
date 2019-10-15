# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import sklearn.model_selection
import sklearn.preprocessing
import sklearn.linear_model
import pandas as pd
import statsmodels.api

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv')

# Categorical data
one_hot = sk.preprocessing.OneHotEncoder(sparse = False, drop = 'first')
state = one_hot.fit_transform(np.expand_dims(data['State'].values, 1))

x, y = np.hstack((data.values[:, :3], state)), data.values[:, -1]
x, y = x.astype(np.float32), y.astype(np.float32)

# Scaling
scaler = sk.preprocessing.StandardScaler()
x = scaler.fit_transform(x)

selected_columns = list(range(len(x[0])))

for i in range(len(selected_columns)):
    x = x[:, selected_columns]
    results = statsmodels.api.OLS(y, x).fit()
    max_p_index = np.argmax(results.pvalues)
    max_p = results.pvalues[max_p_index]
    if max_p < .05:
        break
    del selected_columns[max_p_index]

print('Remaining features are', selected_columns)

