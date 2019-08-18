# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection._split
import statsmodels.api
import pandas as pd

# Load data
data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups - Copy.csv')
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# Categorical data
label_encoder = sk.preprocessing.LabelEncoder()
x[:, 0] = label_encoder.fit_transform(x[:, 0])
one_hot_encoder = sk.preprocessing.OneHotEncoder(categorical_features = [0])
x = one_hot_encoder.fit_transform(x).toarray()

# Dummy variable trap
x = x[:, 1:]

# Feature scaling
scaler = sk.preprocessing.StandardScaler()
x[:, 2:] = scaler.fit_transform(x[:, 2:]) 

# Test - train split
x_train, x_test, y_train, y_test = sk.model_selection._split.train_test_split(x, y, train_size = 0.7)

# Backwords elimination v1
def backwords_elimination(x, y):
    model = statsmodels.api.OLS(exog = x, endog = y).fit()
    r_squared = model.rsquared_adj
    p_values = model.pvalues
    max_p_value = p_values.max()
    max_p_value_index = np.where(p_values == max_p_value)[0][0]
    if max_p_value > 0.05:
        new_x = np.delete(x, max_p_value_index, axis = 1)    
        new_model, selected_x = backwords_elimination(new_x, y)
        new_r_squared = new_model.rsquared_adj
        return (model, x) if r_squared > new_r_squared else (new_model, selected_x)
    else:
        return (model, x)
    
#model, selected_x = backwords_elimination(x_train, y_train)
#model.summary()

# Backwords elimination v2
def backwords_elimination_v2(selected_indices):
    x = x_train[:, selected_indices]
    model = statsmodels.api.OLS(exog = x, endog = y_train).fit()
    r_squared = model.rsquared_adj
    p_values = model.pvalues
    max_p_value = p_values.max()
    max_p_value_index = np.where(p_values == max_p_value)[0][0]
    if max_p_value > 0.05:
        new_indices = selected_indices.copy()
        del new_indices[max_p_value_index]    
        new_model, selected_indices = backwords_elimination_v2(new_indices)
        new_r_squared = new_model.rsquared_adj
        return (model, selected_indices) if r_squared > new_r_squared else (new_model, selected_indices)
    else:
        return (model, selected_indices)
    
model, selected_indices = backwords_elimination_v2(list(range(len(x_train[0]))))
