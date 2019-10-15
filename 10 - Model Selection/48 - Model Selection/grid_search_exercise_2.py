# -*- coding: utf-8 -[*

import numpy as np
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.svm
import pandas as pd

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 10 - Model Selection & Boosting/Section 48 - Model Selection/Social_Network_Ads.csv')

x, y = data.values[:, 1:-1], data.values[:, -1]
label_encoder = sk.preprocessing.LabelEncoder()
x[:, 0] = label_encoder.fit_transform(x[:, 0])
x, y = x.astype(np.float32), y.astype(np.float32)

scaler = sk.preprocessing.StandardScaler()
x = scaler.fit_transform(x)

model = sk.svm.SVC()

grid_serch = sk.model_selection.GridSearchCV(
    estimator = model,
    param_grid = dict( 
            C = [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2], 
            kernel = ['rbf', 'linear', 'poly', 'sigmoid'], 
            degree = [1, 2, 3, 4, 5, 6, 7],  
            gamma = ['auto', 'scale']
    ),
    cv = 10
)
    
results = grid_serch.fit(x, y)

print(results.best_params_)