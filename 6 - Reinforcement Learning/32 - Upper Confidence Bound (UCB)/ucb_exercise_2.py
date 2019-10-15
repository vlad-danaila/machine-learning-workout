# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import matplotlib.pyplot as plt
import pandas as pd
import math
import matplotlib.pyplot as plt

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv')

N = len(data)
nb_selected = np.ones(10)
total_rewards = np.zeros(10)
ratio = 3 / 2 
action = 0

for i, ad in data.iterrows():
    means = total_rewards / nb_selected
    intervals = np.sqrt(ratio * np.log(np.full(10, i + 1)) / nb_selected)
    upper_bounds = means + intervals
    action = np.argmax(upper_bounds)
    nb_selected[action] += 1
    total_rewards[action] += ad[action]
    
plt.bar(range(1, 11), nb_selected)
plt.bar(range(1, 11), total_rewards)
plt.title('Rewards from total nb. of selections')
plt.show()