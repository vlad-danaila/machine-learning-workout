# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import sklearn.decomposition
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.linear_model
import sklearn.discriminant_analysis
import sklearn.metrics
import pandas as pd
import matplotlib.pyplot as plt

DATASET_PATH = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 9 - Dimensionality Reduction/Section 43 - Principal Component Analysis (PCA)/Wine.csv'
data = pd.read_csv(DATASET_PATH)
x, y = data.values[:, :-1], data.values[:, -1]

x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y)

reductions = {
    'pca': sk.decomposition.PCA(n_components = 2),
    'lda': sk.discriminant_analysis.LinearDiscriminantAnalysis(n_components = 2),
    'kpca': sk.decomposition.KernelPCA(n_components = 2, kernel = 'rbf')        
}

for name, reduction in reductions.items():
    x_train = reduction.fit_transform(x_train, y_train) \
        if name == 'lda' else reduction.fit_transform(x_train)
    x_test = reduction.transform(x_test)
    model = sk.linear_model.LogisticRegression(multi_class = 'auto', solver = 'lbfgs')
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    f1 = sk.metrics.f1_score(y_test, y_pred, average = 'weighted')
    print(name, 'F1 socre', f1)
    accuracy = sk.metrics.accuracy_score(y_test, y_pred)
    print(name, 'acuracy', accuracy)
    print()
    