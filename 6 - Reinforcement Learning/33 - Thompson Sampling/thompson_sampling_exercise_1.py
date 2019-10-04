import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats

DATASET_PATH = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv'
ads = pd.read_csv(DATASET_PATH).values

for ad in ads:
    print(ad)
    
beta = scipy.stats.beta(a = 10, b = 10)
beta.pdf([.2, .4, .9])