# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn as sk
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.linear_model
import sklearn.metrics
import sklearn.discriminant_analysis

DATASET_PATH = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 9 - Dimensionality Reduction/Section 43 - Principal Component Analysis (PCA)/Wine.csv'
COLORS = 'red', 'green', 'blue'

data = pd.read_csv(DATASET_PATH)
x, y = data.iloc[:, :-1].values, data.iloc[:, -1].values

# Split
x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y, test_size = .25)

# Scaling
scaler = sk.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Define model
model = sk.linear_model.LogisticRegression()

def make_linspace(x, steps = 100):
    x_min, x_max = min(x), max(x)
    padd = (x_max - x_min) / 5
    return np.linspace(x_min - padd, x_max + padd, steps)

def train_test_visulaize(dim_reduction_name, dim_reduction):
    x_train_transformed = dim_reduction.fit_transform(x_train) \
        if dim_reduction_name != 'LDA' \
        else dim_reduction.fit_transform(x_train, y_train)
    x_test_transformed = dim_reduction.transform(x_test)
    model.fit(x_train_transformed, y_train)
    y_pred = model.predict(x_test_transformed)
    conf_matrix = sk.metrics.confusion_matrix(y_test, y_pred)
    accuracy = np.sum(np.diag(conf_matrix)) / np.sum(conf_matrix)

    x0, x1 = make_linspace(x_train_transformed[:, 0]), make_linspace(x_train_transformed[:, 1])
    x0_mesh, x1_mesh = np.meshgrid(x0, x1)
    x0_1 = np.c_[x0_mesh.ravel(), x1_mesh.ravel()]
    y_grid = model.predict(x0_1)

    cmap = matplotlib.colors.ListedColormap(COLORS)
    plt.contourf(x0, x1, y_grid.reshape(x0_mesh.shape), cmap = cmap, alpha = .2)
    for i in range(1, 4):
        is_class = y_test == i
        plt.scatter(
            x_test_transformed[is_class, 0], 
            x_test_transformed[is_class, 1], 
            color = COLORS[i - 1]
        )
        plt.title('{} (accuracy {})'.format(dim_reduction_name, accuracy))
    plt.show()
        
# Dimensionality reduction techniques
dim_reductions = {}
dim_reductions['PCA'] = sk.decomposition.PCA(2)
dim_reductions['Kernel PCA'] = sk.decomposition.KernelPCA(2, 'rbf')
dim_reductions['LDA'] = sk.discriminant_analysis.LinearDiscriminantAnalysis()

for elem in dim_reductions.items():
    train_test_visulaize(*elem)