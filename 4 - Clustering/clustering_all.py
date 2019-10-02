# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
import sklearn as sk
import sklearn.preprocessing
import sklearn.cluster

DATASET_PATH = 'C:/DOC\Workspace/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 24 - K-Means Clustering/Mall_customers.csv'

data = pd.read_csv(DATASET_PATH)
x = data.iloc[:, [3, 4]].values

colors = 'red', 'green', 'blue', 'black', 'yellow', 'brown', 'orange', 'pink'

def get_color(i):
    return colors[i % len(colors)]

def cluster_variance(cluster):
    return np.linalg.norm(cluster - cluster.mean(axis = 0), axis = 1).sum()

def cluster_data(nb_clusters, cluster_model, model_name):
    hierarchical = cluster_model(nb_clusters)
    clustered = hierarchical.fit_predict(x)
    total_variance = 0
    for i in range(nb_clusters):
        cluster = x[clustered == i]  
        total_variance += cluster_variance(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], color = get_color(i))
    plt.title('{} ({} clusters, {} variance)'.format(model_name, nb_clusters, total_variance))
    return total_variance

variances_k, variances_h = [], [] 

for nb_clusters in range(2, 11):
    fig = plt.figure(figsize = (15, 5))
    fig.add_subplot(1, 2, 1)
    variance_k = cluster_data(nb_clusters, sk.cluster.KMeans, 'KMeans')
    variances_k.append(variance_k)
    fig.add_subplot(1, 2, 2)
    variance_h = cluster_data(nb_clusters, sk.cluster.AgglomerativeClustering, 'Hierarchical')
    variances_h.append(variance_h)
    
def find_boldest_angle(v):
    #torsion = np.abs(np.gradient(np.gradient(np.gradient(v))))
    #return np.argmax(torsion) + 1clear
    slope = np.gradient(v, 1)
    curvature = np.gradient(slope, 1, edge_order = 2)
    torsion = np.gradient(curvature, 1, edge_order = 2)
    torsion_abs = np.abs(torsion)
    return np.argmax(torsion_abs) + 2

    
fig = plt.figure(figsize = (15, 5))

fig.add_subplot(1, 2, 1)
plt.plot(range(2, len(variances_k) + 2), variances_k)
plt.title('Variance KMeans')

fig.add_subplot(1, 2, 2)
plt.plot(range(2, len(variances_h) + 2), variances_h)
best_nb_clusters = find_boldest_angle(variances_h)
plt.scatter(best_nb_clusters, variances_h[best_nb_clusters - 2], color = 'red', s = 100)
plt.title('Variance Hierarchical')