# -*- coding: utf-8 -*-

import numpy as np, math, pandas as pd, matplotlib.pyplot as plt

x = list(range(10, 1000))
y = [ math.sqrt(3/2 * math.log(_x) / math.ceil(_x / 200))  for _x in x]

plt.plot(x, y)
plt.title('Bound fluctuation visualisation')
plt.xlabel('Nr. of rounds')
plt.ylabel('Bound')
plt.show() 

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv')
ads = data.iloc[:].values

N = len(ads[0])
counts, rewards, upper_bounds = np.zeros(N), np.zeros(N), np.full((N), np.Infinity)

for ad in ads:
    print(ad)