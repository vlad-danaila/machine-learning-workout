# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

data = pd.read_csv('C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv')
x = data.values.astype(np.byte)

class ConfidenceBound:
    def __init__(self):
        self.nb_selections = 1e-10
        self.total_reward = 1e-10
    
    def center(self):
        return self.total_reward / self.nb_selections
    
    def bound(self, step):
        return math.sqrt(3/2 * math.log(step + 1) / self.nb_selections)
    
    def upper_bound(self, step):
        return self.center() + self.bound(step)

    def lower_bound(self, step):
        return self.center() - self.bound(step)
            
    def update(self, reward):
        self.nb_selections += 1
        self.total_reward += reward
        
def plot_bounds(bounds, step):
    low = [b.lower_bound(step) for b in bounds]
    segments = [b.upper_bound(step) - b.lower_bound(step) for b in bounds]
    centers = [b.center() for b in bounds]
    plt.bar(range(1, 11), height = segments, bottom = low)
    plt.bar(range(1, 11), height = .01, bottom = centers, color = 'red')
    plt.title('UCB confidence bounds - step {}'.format(step))
    plt.show()

def plt_nb_selections(bounds):
    low = [0 for b in bounds]
    selections = [b.nb_selections for b in bounds]
    plt.bar(range(1, 11), height = selections, bottom = low)
    plt.title('Number of selections')
    plt.show()

bounds = [ConfidenceBound() for i in range(10)]

for i in range(len(x)):
    selected_action = np.array([b.upper_bound(i) for b in bounds]).argmax()
    bounds[selected_action].update(x[i][selected_action])
    if i != 0 and i % 1000 == 0:
        plot_bounds(bounds, i)
        
plt_nb_selections(bounds)