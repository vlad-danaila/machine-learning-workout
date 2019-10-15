# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pylab as plt
import pandas as pd

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv')

def ones_and_a_zero(i):
    v = np.ones(10)
    v[i] = 0
    return v

rewarded, non_rewarded = np.ones(10), np.ones(10)
nb_selections = np.zeros(10)

for i, ad in data.iterrows():
    samples = np.random.beta(rewarded, non_rewarded, 10)
    action = np.argmax(samples)
    reward = ad[action]
    if reward > 0:    
        rewarded[action] += reward
    else:
        non_rewarded[action] += 1
    nb_selections[action] += 1

plt.bar(range(1, 11), nb_selections)
plt.bar(range(1, 11), rewarded)
plt.title('Rewarded out of selected')