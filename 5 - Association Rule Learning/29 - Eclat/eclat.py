# -*- coding: utf-8 -*-

import collections, itertools, pandas as pd

MIN_SUPPORT = 0.001
MIN_LENGTH = 3
MAX_LENGTH = 4

DATA_PATH = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 5 - Association Rule Learning/Section 28 - Apriori/Market_Basket_Optimisation.csv'
data = pd.read_csv(DATA_PATH, header = None)

def get_combinations(v):
    for i in range(MIN_LENGTH, min(MAX_LENGTH + 1, max(MIN_LENGTH, len(v) + 1))):
        for x in itertools.combinations(v, i):
            yield frozenset(x)

total = len(data)
products_support = collections.defaultdict(lambda: 0)

for i, transaction in data.iterrows():
    products = [product for product in transaction if type(product) is str]
    for prod_combination in get_combinations(products):
        products_support[prod_combination] += 1 / total    
        
products_support_list = [ (p, support) for p, support in products_support.items() if support > MIN_SUPPORT ]
products_support_list.sort(key = lambda elem: elem[1], reverse = True)

print('Best 10:')
print(*products_support_list[:10], sep = '\n')
