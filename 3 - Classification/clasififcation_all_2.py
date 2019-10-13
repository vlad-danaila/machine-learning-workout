import numpy as np
import sklearn as sk
import sklearn.preprocessing
import sklearn.model_selection
import sklearn.metrics
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.linear_model
import sklearn.svm
import sklearn.naive_bayes
import sklearn.tree
import sklearn.ensemble

dataset_path = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)/Social_Network_Ads.csv'
data = pd.read_csv(dataset_path)

# Data preprocessing
x = data.values[:, 2:-1]
y = data.values[:, -1]
x, y = x.astype(np.float32), y.astype(np.float32)

x_train, x_test, y_train, y_test = sk.model_selection.train_test_split(x, y)
scaler = sk.preprocessing.StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

models = {}
models['logistic'] = sk.linear_model.LogisticRegression()
#models['knn'] = sk.neighbors.KNeighborsClassifier()
models['tree'] = sk.tree.DecisionTreeClassifier()
models['forest'] = sk.ensemble.RandomForestClassifier()
models['bayes'] = sk.naive_bayes.GaussianNB()
models['svm'] = sk.svm.SVC()

def make_grid(x, steps = 1000):
    x_min, x_max = min(x), max(x)
    diff = x_max - x_min
    padd = diff / 10
    return np.linspace(x_min - padd, x_max + padd, steps)

def plot_decision_boundry(x, y, model, name, accuracy):
    x0, x1 = make_grid(x[:, 0]), make_grid(x[:, 1])
    mesh_x0, mesh_x1 = np.meshgrid(x0, x1)
    x01 = np.c_[mesh_x0.ravel(), mesh_x1.ravel()]
    print(x01.shape)
    y_grid = model.predict(x01)
    plt.contourf(x0, x1, y_grid.reshape(mesh_x0.shape))
    plt.scatter(x[y == 0, 0], x[y == 0, 1])
    plt.scatter(x[y == 1, 0], x[y == 1, 1])
    plt.title('{} ({})'.format(name, accuracy))
    plt.show()
    
for name, model in models.items():
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    cm = sk.metrics.confusion_matrix(y_pred, y_test)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    plot_decision_boundry(x_test, y_test, model, name, accuracy)