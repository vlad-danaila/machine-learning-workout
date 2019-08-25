# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pylab as plt
import sklearn.cluster.hierarchical
import scipy.cluster.hierarchy

data = pd.read_csv('C:/DOC\Workspace/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 24 - K-Means Clustering/Mall_customers.csv')
x = data.iloc[:, [3, 4]].values

# Plot dendogram
linked = scipy.cluster.hierarchy.linkage(x, method='ward') 
dendogram = scipy.cluster.hierarchy.dendrogram(linked)
plt.title('Dendogram')
plt.ylabel('Cluster distance')
plt.xlabel('Clusters')
plt.show()

# Fit model
n_clusters = 5
clustering = sklearn.cluster.hierarchical.AgglomerativeClustering(n_clusters = n_clusters)
x_clustered = clustering.fit_predict(x)

# Plot clusters
colors = ('red', 'green', 'blue', 'yellow', 'orange')
get_color = lambda i: colors[i % len(colors)]

for i in range(n_clusters):
    plt.scatter(x[x_clustered == i, 0], x[x_clustered == i, 1], color = get_color(i))
    # Plot centroids
    plt.scatter(np.mean(x[x_clustered == i, 0]), np.mean(x[x_clustered == i, 1]), color = 'black')
plt.title('Clusters')
plt.xlabel('Income')
plt.ylabel('Spending score')
plt.show()