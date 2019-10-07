# -*- coding: utf-8 -*-

import pandas as pd
import collections
import itertools

COMBINATIONS_MAX_SIZE = 3
MIN_PROBABILITY = 1e-2

DATA_PATH = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 5 - Association Rule Learning/Section 28 - Apriori/Market_Basket_Optimisation.csv'
data = pd.read_csv(DATA_PATH, header = None)

N = len(data)
counts = collections.defaultdict(lambda: 0)

def combinations(x, size = COMBINATIONS_MAX_SIZE):
    for combinations_size in range(1, size + 1):
        for combination in itertools.combinations(x, combinations_size):
            yield combination

for products in data.values:
    products = [p for p in products if type(p) == str]
    for combination in combinations(products):
        counts[frozenset(combination)] += 1

probabilitites = {}

for key, count in counts.items():
    probability = count / N
    if probability > MIN_PROBABILITY:
        probabilitites[key] = probability
        
lifts = {}

for group, probability in probabilitites.items():
    if len(group) == 1:
        continue
    for perm in itertools.permutations(combinations(group), 2):
        diff = frozenset(perm[1]) - frozenset(perm[0])
        if len(diff) == 0:
            continue
        lift_params = frozenset(perm[0]), diff
        inv_lift_params = lift_params[1], lift_params[0] 
        if lift_params in lifts or inv_lift_params in lifts:
            continue
        lifts[lift_params] = probabilitites[group] / (probabilitites[lift_params[0]] * probabilitites[lift_params[1]])
        
values_of_lifts = { v: k for k, v in lifts.items() }
          
print('LIFTS', '=' * 100)
for lift in sorted(values_of_lifts.items(), reverse = True):
    print(lift[0], ':', lift[1])