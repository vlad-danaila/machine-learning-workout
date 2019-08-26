# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.2
MIN_LIFT = 3

DATA_PATH = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 5 - Association Rule Learning/Section 28 - Apriori/Market_Basket_Optimisation.csv'
data = pd.read_csv(DATA_PATH, header = None)

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

# transactions variable will keep the fitered and cleaned up data
transactions = []

# Count elements for calculating support
products_count = {}
for i, row in data.iterrows(): # for each transaction
    transaction = set()
    for product in row: # for each product in the transaction
        if type(product) is str: # if valid product
            transaction.add(product)
            if product in products_count: # do the counting
                products_count[product] += 1
            else:
                products_count[product] = 1
    transactions.append(transaction)        

# Calculate support
total = len(data)
products_support = { k : v / total for k, v in products_count.items() if v / total > MIN_SUPPORT }

# Count combinations
prod_pair_count = {}
for main_product in products_support.keys(): # iterate through products that have a support higher then treshold
    for transaction in transactions: # iterate through the transactions
        if main_product in transaction: 
            for product in transaction:
                if product in products_support and product != main_product:
                    product_pair = main_product, product
                    if product_pair in prod_pair_count:
                        prod_pair_count[product_pair] += 1
                    else:
                        prod_pair_count[product_pair] = 1


# Calculate confidence
confidence = { products : (count / products_count[products[1]]) for products, count in prod_pair_count.items() 
    if (count / products_count[products[1]]) > MIN_CONFIDENCE 
}

# Calculate lift
lift = { products : (conf / products_support[products[0]]) for products, conf in confidence.items() 
    if (conf / products_support[products[0]]) > MIN_LIFT and products_support[products[0]] > MIN_SUPPORT }

sorted_lifts = sorted(lift.items(), key = lambda elem: elem[1], reverse = True)