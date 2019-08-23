# -*- coding: utf-8 -*-

import numpy as np
import sklearn.preprocessing
import sklearn as sk
import sklearn.cluster
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('C:/DOC\Workspace/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 24 - K-Means Clustering/Mall_customers.csv')
x = data.iloc[:, [3,4]].values

fig, sublot = plt.subplots(2)
fig.suptitle('K-Means')

# Within cluster sum of squares (WCSS)
wcss = []
max_nr_clusters = 20
for nr_clusters in range(2, max_nr_clusters):
    k_means = sk.cluster.KMeans(n_clusters = nr_clusters)    
    k_means.fit(x)
    wcss.append(k_means.inertia_)
    
sublot[0].plot(range(2, max_nr_clusters), wcss)
sublot[0].set_title('Within cluster sum of squares')

# Cluster
nr_clusters = 5
k_means = sk.cluster.KMeans(n_clusters = nr_clusters)
x_clustered = k_means.fit_predict(x)

# Plot clusters
for _class, color in enumerate(('red', 'green', 'blue', 'yellow', 'orange')):
    is_class = x_clustered == _class
    sublot[1].scatter(x[is_class, 0], x[is_class, 1], color = color)
plt.show()