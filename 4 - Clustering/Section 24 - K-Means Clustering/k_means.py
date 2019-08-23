# -*- coding: utf-8 -*-

import math
import numpy as np
import sklearn.preprocessing
import sklearn as sk
import sklearn.cluster
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('C:/DOC\Workspace/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 24 - K-Means Clustering/Mall_customers.csv')
x = data.iloc[:, [3,4]].values

fig, subplots = plt.subplots(2, figsize = (10, 10))
fig.suptitle('K-Means')
plt.subplots_adjust(hspace = 0.5)

# Within cluster sum of squares (WCSS)
wcss = []
max_nr_clusters = 20
for nr_clusters in range(2, max_nr_clusters):
    k_means = sk.cluster.KMeans(n_clusters = nr_clusters)    
    k_means.fit(x)
    wcss.append(k_means.inertia_)
    
subplots[0].plot(range(2, max_nr_clusters), wcss)
subplots[0].set_title('Within cluster sum of squares')
subplots[0].set_xlabel('Nr. Clusters')
subplots[0].set_ylabel('WCSS')

# Find the best nr of clusters
slope = np.gradient(wcss)
curvature = np.gradient(slope)
torsion = np.gradient(curvature)
torsion_abs = np.abs(torsion)
nr_clusters = np.where(torsion_abs == torsion_abs.max())[0][0] + 2
print('The best number of clusters is', nr_clusters)
subplots[0].scatter(nr_clusters, wcss[nr_clusters - 2], color = 'red', s = 100)

# Cluster
k_means = sk.cluster.KMeans(n_clusters = nr_clusters)
x_clustered = k_means.fit_predict(x)

# Plot clusters
colors = ('red', 'green', 'blue', 'yellow', 'pink', 'orange')

def get_color(i):
    return colors[i % len(colors)]

for i in range(nr_clusters):
    is_class = x_clustered == i
    subplots[1].scatter(x[is_class, 0], x[is_class, 1], color = get_color(i))
centroids = k_means.cluster_centers_
subplots[1].scatter(centroids[:, 0], centroids[:, 1], color = 'black', s = 150)
subplots[1].set_title('Clusters')
subplots[1].set_xlabel('Inconme')
subplots[1].set_ylabel('Spending Score')
plt.show()


