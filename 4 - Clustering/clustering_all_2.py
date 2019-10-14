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

def cluster_spread(n_clusters, labels):
    groups = [ x[labels == i] for i in range(n_clusters) ]
    centroids = np.array([ np.mean(groups[i], axis = 0) for i in range(n_clusters) ])
    spreads = [ np.sum(np.linalg.norm((groups[i] - centroids[i]) ** 2, axis = 1)) for i in range(n_clusters) ]
    return centroids, np.sum(spreads) / len(x)

def cluster_k_means(n_clusters):
    _, labels, inertia = sk.cluster.k_means(x, n_clusters = n_clusters)
    centroids, spread = cluster_spread(n_clusters, labels)
    plot_cluster(n_clusters, labels, centroids, 'Kmeans')
    return spread

def cluster_hierarchical(n_clusters):
    hierarchical = sk.cluster.hierarchical.AgglomerativeClustering(n_clusters)
    labels = hierarchical.fit_predict(x)
    centroids, spread = cluster_spread(n_clusters, labels)
    plot_cluster(n_clusters, labels, centroids, 'Hierarchical')
    return spread
        
def plot_cluster(n_clusters, labels, centroids, title):
    for cluster_class in range(n_clusters):
        plt.scatter(x[labels == cluster_class, 0], x[labels == cluster_class, 1])
        plt.scatter(centroids[cluster_class, 0], centroids[cluster_class, 1], color = 'black', s = 200)
        plt.title(title)
    plt.show()
    
def plot_spread_vs_n_clusters(cluster_spread, best_nb_clusters, method):
    plt.plot(range(1, len(cluster_spread) + 1), cluster_spread)
    plt.title('Cluster Spread vs. Nb. Clusters ({})'.format(method))
    plt.xlabel('Nb. Clusters')
    plt.ylabel('Spread')
    plt.scatter(best_nb_clusters, cluster_spread[best_nb_clusters - 1])
    
def to_angle(v):
    return cmath.phase(v) / cmath.pi * 180    
    
def best_nb_clusters(cluster_spread):
    angles = []
    for i in range(len(cluster_spread) - 2):
        spread1, spread2, spread3 = cluster_spread[i], cluster_spread[i + 1], cluster_spread[i + 2]
        v1 = complex(-1, (spread1 - spread2) / spread3) 
        v2 = complex(1, (spread3 - spread2) / spread3)
        angles.append(to_angle(v1) - to_angle(v2))
    best_nb_clusters = np.argmin(np.array(angles)) + 2
    return best_nb_clusters

cluster_spread_k, cluster_spread_h = [], []      

for n_clusters in range(1, 11):
    cluster_spread_k.append(cluster_k_means(n_clusters))
    cluster_spread_h.append(cluster_hierarchical(n_clusters))
    
plot_spread_vs_n_clusters(cluster_spread_k, best_nb_clusters(cluster_spread_k), 'K Means')
plot_spread_vs_n_clusters(cluster_spread_h, best_nb_clusters(cluster_spread_h), 'Hierarchical')