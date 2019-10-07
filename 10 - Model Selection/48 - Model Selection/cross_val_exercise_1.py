# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.svm

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 10 - Model Selection & Boosting/Section 48 - Model Selection/Social_Network_Ads.csv')

x, y = data.values[:, 1:-1], data.values[:, -1]

# Categorical data handling
one_hot = sk.preprocessing.OneHotEncoder(sparse = False, drop = 'first')
x[:, 0] = np.squeeze(one_hot.fit_transform(x[:, np.newaxis, 0]))
x, y = x.astype(np.float32), y.astype(np.float32)

# Scaling
scaler = sk.preprocessing.StandardScaler()
x = scaler.fit_transform(x)

# Define model
model = sk.svm.SVC(gamma = 'auto')

scores = sk.model_selection.cross_validate(
        model, x, y, scoring = ('accuracy', 'precision', 'recall', 'f1'), cv = 10, error_score = 'raise')

print(scores)
