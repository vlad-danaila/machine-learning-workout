import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

DATASET_PATH = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv'
ads = pd.read_csv(DATASET_PATH).values

def plot():
    plt.hist(selected)
    plt.title('Nb. selections ({} iterations)'.format(i + 1))
    plt.show()

N = len(ads[0])
rewards = np.ones(10), np.ones(10)
selected = []

for i in range(len(ads)):
    beta = np.random.beta(rewards[1], rewards[0], 10)
    action = np.argmax(beta)    
    reward = ads[i][action]
    rewards[reward][action] += 1
    selected.append(action)
    if i % 1000 == 0:
        plot()

plot()