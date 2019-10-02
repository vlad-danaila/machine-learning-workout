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

def cluster_kmeans(nb_clusters):
    kmeans = sk.cluster.KMeans(nb_clusters)
    clustered = kmeans.fit_predict(x)
    for i in range(nb_clusters):
        plt.scatter(x[clustered == i, 0], x[clustered == i, 1], color = get_color(i))
    plt.title('KMeans ({} clusters, {} variance)'.format(nb_clusters, kmeans.inertia_))
    plt.show()
    return kmeans.inertia_

#for nb_clusters in range(1, 11):
 #   cluster_kmeans(nb_clusters)

for nb_clusters in range(1, 11):
    hierarchical = sk.cluster.AgglomerativeClustering(nb_clusters)
    clustered = hierarchical.fit_predict(x)
    total_variance = 0
    for i in range(nb_clusters):
        cluster = x[clustered == i]  
        total_variance += cluster_variance(cluster)
        plt.scatter(cluster[:, 0], cluster[:, 1], color = get_color(i))
    plt.title('Hierarchical ({} clusters, {} variance)'.format(nb_clusters, total_variance))
    plt.show()

    

        

