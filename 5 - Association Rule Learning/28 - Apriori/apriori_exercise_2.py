# -*- coding: utf-8 -*-

import numpy as np
import sklearn as sk
import pandas as pd
import matplotlib.pyplot as plt
import itertools
import collections

max_len_combinations = 3
prods_set_probability_treshold = .01


DATA_PATH = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 5 - Association Rule Learning/Section 28 - Apriori/Market_Basket_Optimisation.csv'
data = pd.read_csv(DATA_PATH, header = None)

def get_products():
    for i, transaction in data.iterrows():
        yield [prod for prod in transaction if type(prod) == str]
    
def get_combinations(products, max_size = max_len_combinations):
    for size in range(1, max_size + 1):
        for comb in itertools.combinations(products, size):
            yield frozenset(comb)        

def get_prod_probabilities():
    prod_sets = collections.defaultdict(lambda: 0)        
    for products in get_products():
        for combination in get_combinations(products):
            prod_sets[combination] += 1
    N = len(data)
    probabilities = {}
    for prods, count in prod_sets.items():
        if count / N > prods_set_probability_treshold:
            probabilities[prods] = count / N
    return probabilities

def lift(prods_1, prods_2):
    p1 = probabilitites[prods_1]
    p2 = probabilitites[prods_2]
    p1_p2 = probabilitites[prods_1.union(prods_2)]
    return p1_p2 / (p1 * p2)

probabilitites = get_prod_probabilities()

lifts = {}

for prob in probabilitites:
    if len(prob) > 1:
        comb = get_combinations(prob, len(prob) - 1)
        for pair in itertools.combinations(comb, 2):
            intersect = pair[0].intersection(pair[1])
            if len(intersect) > 0:
                p1 = pair[0].difference(intersect)
                p2 = pair[1].difference(intersect)
                if len(p1) == 0 or len(p2) == 0:
                    continue
                pair = p1, p2
            lifts[lift(*pair)] = pair
        
for lift, pair in sorted(lifts.items(), reverse = True):
    print(lift, list(pair[0]), list(pair[1]))
    
