# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.svm

# Data loading
data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 10 - Model Selection & Boosting/Section 48 - Model Selection/Social_Network_Ads.csv')
x, y = data.iloc[:, 1:-1].values, data.iloc[:, -1].values

# Handle categorical data
label_encoder = sk.preprocessing.LabelEncoder()
x[:, 0] = label_encoder.fit_transform(x[:, 0])

x = x.astype(np.float64)

# Scale data
scaler = sk.preprocessing.StandardScaler()
x[:, 1:] = scaler.fit_transform(x[:, 1:])

# Fit SVM classifier
svm = sk.svm.SVC()

# Cross validation
recalls = sk.model_selection.cross_val_score(svm, x, y, cv = 10, scoring = 'recall')
print('Recall is', recalls.mean(), 'with a deviation of', recalls.std())

f1_scores = sk.model_selection.cross_val_score(svm, x, y, cv = 10, scoring = 'f1')
print('F1 score is', f1_scores.mean(), 'with a deviation of', f1_scores.std())