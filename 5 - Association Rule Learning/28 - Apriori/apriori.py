# -*- coding: utf-8 -*-

import math
import numpy as np
import pandas as pd

support_treshold = 0.01
confidence_treshold = 0.2
lift_treshold = 3

data_path = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 5 - Association Rule Learning/Section 28 - Apriori/Market_Basket_Optimisation.csv'
data = pd.read_csv(data_path, header = None)

'''
1. Initialize a map for support
2. Iterate through the dataset, calculate support for each element and store in map.
3. Keep only a part of elements with support higher then x
4. For those elements, calculate confidence 
5. Filter by confidence treshold
6. Calculate lift
7. Filter by lift trehold
8. Sort by lift
'''

# Count elements
count_elems = {}
for i, elems in data.iterrows():
    for e in elems:
        if type(e) is str:
            if e in count_elems.keys():
                count_elems[e] += 1
            else:
                count_elems[e] = 1

# Calculate support
total = len(data)
support = { key : (value / total) for key, value in count_elems.items() if (value / total) >= support_treshold }

# Count combinations
comb_elems = {}
for support_elem in support.keys():
    for i, elems in data.iterrows():
        if support_elem in set(elems):
            combinations = [ (support_elem, e) for e in elems if type(e) is str and e != support_elem ]    
            for comb in combinations:
                if comb in comb_elems.keys():
                    comb_elems[comb] += 1
                else:
                    comb_elems[comb] = 1

# Calculate confidence
confidence = { products : (count / count_elems[products[1]]) for products, count in comb_elems.items() 
    if (count / count_elems[products[1]]) >= confidence_treshold }

# Calculate lift
lift = { products : (conf / support[products[0]]) for products, conf in confidence.items() 
    if (conf / support[products[0]]) > lift_treshold }

sorted_lifts = sorted(lift.items(), key = lambda elem: elem[1], reverse = True)