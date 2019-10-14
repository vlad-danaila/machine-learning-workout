# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import sklearn.cluster
import matplotlib.pyplot as plt
import pandas as pd
import cmath

data_path = 'C:/DOC\Workspace/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 24 - K-Means Clustering/Mall_customers.csv'
data = pd.read_csv(data_path)

x = data.values[:, [3, 4]].astype(np.float32)

def cluster_k_means(n_clusters):
    centroids, labels, inertia = sk.cluster.k_means(x, n_clusters = n_clusters)
    plot_cluster(n_clusters, labels, centroids, 'Kmeans')
    return inertia

def cluster_hierarchical(n_clusters):
    hierarchical = sk.cluster.hierarchical.AgglomerativeClustering(n_clusters)
    labels = hierarchical.fit_predict(x)
    centroids = np.array([ np.mean(x[labels == i], axis = 0) for i in range(n_clusters) ])
    plot_cluster(n_clusters, labels, centroids, 'Hierarchical')
        
def plot_cluster(n_clusters, labels, centroids, title):
    print(centroids)
    for cluster_class in range(n_clusters):
        plt.scatter(x[labels == cluster_class, 0], x[labels == cluster_class, 1])
        plt.scatter(centroids[cluster_class, 0], centroids[cluster_class, 1], color = 'black', s = 200)
        plt.title(title)
    plt.show()
    
cluster_spread = []    
angles = []    

for n_clusters in range(1, 11):
    cluster_spread.append(cluster_k_means(n_clusters))
    
for n_clusters in range(1, 11):
    cluster_hierarchical(n_clusters)

def to_angle(v):
    return cmath.phase(v) / cmath.pi * 180
    
for i in range(len(cluster_spread) - 2):
    spread1, spread2, spread3 = cluster_spread[i], cluster_spread[i + 1], cluster_spread[i + 2]
    v1 = complex(-1, (spread1 - spread2) / spread3) 
    v2 = complex(1, (spread3 - spread2) / spread3)
    angles.append(to_angle(v1) - to_angle(v2))
    
best_nb_clusters = np.argmin(np.array(angles)) + 2
 
plt.plot(range(1, len(cluster_spread) + 1), cluster_spread)
plt.title('Cluster Spread vs. Nb. Clusters')
plt.xlabel('Nb. Clusters')
plt.ylabel('Spread')
plt.scatter(best_nb_clusters, cluster_spread[best_nb_clusters - 1])

