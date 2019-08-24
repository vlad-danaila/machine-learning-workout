# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pylab as plt
import sklearn.cluster
import scipy.cluster.hierarchy

data = pd.read_csv('C:/DOC\Workspace/Machine Learning A-Z Template Folder/Part 4 - Clustering/Section 24 - K-Means Clustering/Mall_customers.csv')
x = data.iloc[:, [3, 4]].values

# Plot dendogram
linked = scipy.cluster.hierarchy.linkage(x, method='ward') 
dendogram = scipy.cluster.hierarchy.dendrogram(linked)
plt.show()