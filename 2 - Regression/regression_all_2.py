# -*- coding: utf-8 -*-
import numpy as np
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.linear_model
import sklearn.tree
import sklearn.ensemble
import sklearn.svm
import sklearn.decomposition
import pandas as pd
import matplotlib.pyplot as plt

dataset_path = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv'
data = pd.read_csv(dataset_path)

# Data preprocessing
one_hot = sk.preprocessing.OneHotEncoder(sparse = False, drop = 'first')
country = one_hot.fit_transform(np.expand_dims(data['State'].values, 1))
x = np.hstack((data.values[:, :-2], country))
y = data.values[:, -1]
x, y = x.astype(np.float32), y.astype(np.float32)

x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y)

scaler_x = sk.preprocessing.StandardScaler()
scaler_y = sk.preprocessing.StandardScaler()
x_train_scaled = scaler_x.fit_transform(x_train)
x_test_scaled = scaler_x.transform(x_test)
y_train_scaled = scaler_y.fit_transform(np.expand_dims(y_train, 1)).reshape(-1)
y_test_scaled = scaler_y.transform(np.expand_dims(y_test, 1)).reshape(-1)

pca = sk.decomposition.PCA(n_components = 1)
x_train_pca = pca.fit_transform(x_train_scaled)
x_test_pca = pca.transform(x_test_scaled)

models = {
    'linear': sk.linear_model.LinearRegression(),
    'tree': sk.tree.DecisionTreeRegressor(),
    'forest': sk.ensemble.RandomForestRegressor(),
    'svr': sk.svm.SVR()
}

for name, model in models.items():
    svr = name == 'svr'
    model.fit(x_train_scaled, y_train_scaled if svr else y_train)
    y_pred = model.predict(x_test_scaled)
    error = sk.metrics.mean_absolute_error(
            scaler_y.inverse_transform(y_pred) if svr else y_pred, y_test)
    x_granular = np.linspace(min(x_train_pca) - 1, max(x_train_pca) + 1, 1000)
    y_granular = model.predict(pca.inverse_transform(x_granular))
    plt.plot(x_granular, y_granular)
    plt.scatter(x_test_pca, y_test_scaled if svr else y_test)
    plt.title('{} ({})'.format(name, error))
    plt.xlabel('PCA embedding')
    plt.ylabel('Profit')
    plt.show()