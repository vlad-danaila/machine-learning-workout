# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import xgboost as xg
import sklearn.metrics
import sklearn.decomposition
import matplotlib.pyplot as plt

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 10 - Model Selection & Boosting/Section 49 - XGBoost/Churn_Modelling.csv')

one_hot = sk.preprocessing.OneHotEncoder(sparse = False, drop = 'first')
georgaphy = one_hot.fit_transform(np.expand_dims(data['Geography'], 1))
gender = one_hot.fit_transform(np.expand_dims(data['Gender'], 1))

x, y = data.values[:, 3:-1], data.values[:, -1]
x = np.delete(x, (1, 2), axis = 1)

x = np.hstack((x, georgaphy, gender))

x, y = x.astype(np.float32), y.astype(np.float32)

scaler = sk.preprocessing.StandardScaler()
x = scaler.fit_transform(x)

x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y)

model = xg.XGBClassifier()

model.fit(x_train, y_train)

y_pred = model.predict(x_test)

accuracy = sk.metrics.accuracy_score(y_test, y_pred)
precision = sk.metrics.precision_score(y_test, y_pred)
recall = sk.metrics.recall_score(y_test, y_pred)

print('Accuracy is', accuracy)
print('Precision is', precision)
print('Recall is', recall)

def make_grid(x, steps = 1000):
    x_min, x_max = min(x), max(x)
    diff = x_max - x_min
    padd = diff / 5
    return np.linspace(x_min - padd, x_max + padd, steps)

pca = sk.decomposition.pca.PCA(n_components = 2)
x_pca = pca.fit_transform(x)

x0, x1 = make_grid(x_pca[:, 0]), make_grid(x_pca[:, 1])
mesh_x0, mesh_x1 = np.meshgrid(x0, x1)
x01 = np.c_[mesh_x0.ravel(), mesh_x1.ravel()]
y_grid = model.predict(pca.inverse_transform(x01))

plt.contourf(x0, x1, y_grid.reshape(mesh_x0.shape))
x_test_pca = pca.transform(x_test)
plt.scatter(x_test_pca[y_test == 0, 0], x_test_pca[y_test == 0, 1], s = 1)
plt.scatter(x_test_pca[y_test == 1, 0], x_test_pca[y_test == 1, 1], s = 1)