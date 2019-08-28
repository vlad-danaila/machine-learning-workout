# -*- coding: utf-8 -*-

import numpy as np, random, pandas as pd, matplotlib.pyplot as plt, math, scipy.stats

# Visualize pdf for a few beta distributions
b = 10_000
x = np.linspace(0, .6, 10_000)
for a in range(1000, 10_001, 1000):
    distribution = scipy.stats.beta(a, b)
    plt.scatter(x, distribution.pdf(x), s = 1, label = 'alfa = {}, beta = {}'.format(a, b))
plt.title('Beta PDF')
plt.ylabel('PDF')
plt.xlabel('Random variable value')
plt.legend(bbox_to_anchor=(1, 1))
plt.show()

# Thomson sampling
data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv')
data = data.iloc[:].values

N = len(data[0])
ads_selected = []
ads_count = [np.ones(N), np.ones(N)]

for cliked in data:
    samples = np.array([random.betavariate(ads_count[1][i], ads_count[0][i]) for i in range(N)])
    sample = np.argmax(samples)
    ads_selected.append(sample)
    ads_count[cliked[sample]][sample] += 1 
    
# Plot ads histogram
plt.hist(ads_selected)
plt.title('Ads selected')
plt.xlabel('Ads indices')
plt.ylabel('Nr of selections')
plt.show()