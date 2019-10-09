# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.mmetrics
import sklearn.svm


data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 10 - Model Selection & Boosting/Section 48 - Model Selection/Social_Network_Ads.csv')
x, y = data.values[:, 1 : -1], data.values[:, -1]

# Preprocessing

# Categorical data
one_hot = sk.preprocessing.OneHotEncoder(drop = 'first', sparse = False)
gender = one_hot.fit_transform(x[:, np.newaxis, 0])
x = np.hstack((x[:, 1:], gender))
x, y = x.astype(np.float32), y.astype(np.float32)

# Scaling
scaler = sk.preprocessing.StandardScaler()
x[:, 0:2] = scaler.fit_transform(x[:, 0:2])

# Grid search
grid = sk.model_selection.GridSearchCV(
        sk.svm.SVC(),
        {
            'C': [0.001, 0.01, 0.1, 1, 10],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'gamma': ['auto', 'scale']
        },
        ('accuracy', 'precision', 'recall', 'f1'),
        cv = 10,
        refit =  'f1'
)
grid.fit(x, y)

print(grid.best_params_)
print(grid.best_score_)

model = grid.best_estimator_
y_pred = model.predict(x)
print(sk.metrics.confusion_matrix(y_pred, y))