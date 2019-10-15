# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.svm
import pandas as pd

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 10 - Model Selection & Boosting/Section 48 - Model Selection/Social_Network_Ads.csv')

label_encoder = sk.preprocessing.LabelEncoder()
x, y = data.values[:, 1:-1], data.values[:, -1]
x[:, 0] = label_encoder.fit_transform(x[:, 0])
x, y = x.astype(np.float32), y.astype(np.float32)

scaler = sk.preprocessing.StandardScaler()
x = scaler.fit_transform(x)

model = sk.svm.SVC(gamma = 'auto')

cross_valid = sk.model_selection.cross_validate(
        model, x, y, scoring = ['accuracy', 'precision', 'recall', 'f1'], cv = 10)

print('Average F1 score is', np.average(cross_valid['test_f1']))

