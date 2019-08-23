# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import sklearn.preprocessing
import sklearn.tree
import sklearn.ensemble
import sklearn.model_selection._split
import matplotlib.pyplot as plt
import matplotlib.colors
import pandas as pd
import sklearn.metrics

# Load data
data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 3 - Classification/Section 16 - Support Vector Machine (SVM)/Social_Network_Ads.csv')
x, y = data.iloc[:, 2:-1].values, data.iloc[:, -1].values

# Split data
x_train, x_test, y_train, y_test = sk.model_selection._split.train_test_split(x, y, train_size = 0.7)

def make_grid(x, steps = 100):
    x_min, x_max = x.min(), x.max()
    diff = x_max - x_min
    padding = diff / 4
    return np.linspace(x_min - padding, x_max + padding, steps)

def predict_with_model(model, plot):
    # Fit model
    model.fit(x_train, y_train)
    
    # Predictions    
    y_pred = model.predict(x_test)
    
    # Confusion matrix
    cm = sk.metrics.confusion_matrix(y_test, y_pred)
    accuracy = np.sum(np.diag(cm)) / np.sum(cm)
    print(accuracy)

    # Make grid for plotting
    grid_x_0, grid_x_1 = make_grid(x_test[:, 0]), make_grid(x_test[:, 1])
    grid_mesh_x_0, grid_mesh_x_1 = np.meshgrid(grid_x_0, grid_x_1)
    grid_mesh_x_0_1 = np.array([grid_mesh_x_0.ravel(), grid_mesh_x_1.ravel()]).T
    grid_y = model.predict(grid_mesh_x_0_1).reshape(grid_mesh_x_0.shape)

    # Display plot
    color_map = matplotlib.colors.ListedColormap(('red', 'green'))
    plot.contourf(grid_x_0, grid_x_1, grid_y, cmap = color_map, alpha = 0.2)
    for _class in ((0, 'red'), (1, 'green')):
        plot.scatter(x_test[y_test == _class[0], 0], x_test[y_test == _class[0], 1], color = _class[1])
    plot.set_xlabel('Age')    
    plot.set_ylabel('Salary')

classifier_tree, classifier_forest = sk.tree.DecisionTreeClassifier(), sk.ensemble.RandomForestClassifier(n_estimators = 500)
fig, axes = plt.subplots(2, figsize = (10, 10))
fig.suptitle('Decision tree vs forest')
plt.subplots_adjust(hspace = 0.3)
axes[0].set_title('Decision tree')
axes[1].set_title('Random forest')
predict_with_model(classifier_tree, axes[0])
predict_with_model(classifier_forest, axes[1])
plt.show()




