# -*- coding: utf-8 -*-

import numpy as np, math, pandas as pd, matplotlib.pyplot as plt

x = list(range(10, 1000))
y = [math.sqrt(3/2 * math.log(_x) / math.ceil(_x / 200))  for _x in x]

plt.plot(x, y)
plt.title('Bound fluctuation visualisation')
plt.xlabel('Nr. of rounds')
plt.ylabel('Bound')
plt.show() 

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv')
data = data.iloc[:].values

N = len(data[0])
counts, rewards, mean_rewards = np.zeros(N), np.zeros(N), np.zeros(N)
upper_bounds = np.full((N), np.Infinity)
selected_ads = []

for i, clicked_ads in enumerate(data):
    selected_ad = np.argmax(upper_bounds)
    selected_ads.append(selected_ad)
    counts[selected_ad] += 1
    rewards[selected_ad] += clicked_ads[selected_ad]
    mean_rewards[selected_ad] = rewards[selected_ad] / counts[selected_ad]
    # update bounds
    confidence_bound = math.sqrt(3/2 * math.log(i + 1) / counts[selected_ad])
    upper_bounds[selected_ad] = mean_rewards[selected_ad] + confidence_bound
    
plt.hist(selected_ads)
plt.title('Selected ads')
plt.xlabel('Ads identifiers')
plt.ylabel('Nr. of selections')
plt.show()