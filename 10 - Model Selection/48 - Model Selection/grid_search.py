# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import pandas as pd
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.svm
import sklearn.metrics

# Data loading
data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 10 - Model Selection & Boosting/Section 48 - Model Selection/Social_Network_Ads.csv')
x, y = data.iloc[:, 1:-1].values, data.iloc[:, -1].values

# Handling categorical data
label_encoder = sk.preprocessing.LabelEncoder()
x[:, 0] = label_encoder.fit_transform(x[:, 0])
x = x.astype(np.float64)

# Feature scaling
scaler = sk.preprocessing.StandardScaler()
x[:, 1:] = scaler.fit_transform(x[:, 1:])

# Grid search
grid_search_params = [
                { 
                        'C': [1, 10, 100], 
                        'kernel': ['linear'] 
                },
                { 
                        'C': [1, 10, 100], 
                        'kernel': ['poly', 'rbf', 'sigmoid'], 
                        'gamma': [i/10 for i in range(1, 10)] 
                }
        ]
grid_search = sk.model_selection.GridSearchCV(
        estimator = sk.svm.SVC(),
        param_grid = grid_search_params,
        scoring = 'f1',
        n_jobs = -1,
        cv = 10
)

grid_search.fit(x, y)

# See results
print('Best F1 score is:', grid_search.best_score_)
print('Best params are:', grid_search.best_params_)

# Use fownd model
classifier = grid_search.best_estimator_
y_pred = classifier.predict(x)
cm = sk.metrics.confusion_matrix(y, y_pred)
print(cm)

