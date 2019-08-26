# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

MIN_SUPPORT = 0.01
MIN_CONFIDENCE = 0.2
MIN_LIFT = 3

DATA_PATH = 'C:/DOC/Workspace/Machine Learning A-Z Template Folder/Part 5 - Association Rule Learning/Section 28 - Apriori/Market_Basket_Optimisation.csv'
data = pd.read_csv(DATA_PATH, header = None)

'''
1. Count the elements in order to calculate support
2. Count product pairs in order to calculate confidence
3. Once we have the confidence and support calculate the lift
4. Sort according to lift and display results
'''

# transactions variable will keep the fitered and cleaned up data
transactions = []

# Count elements for calculating support
products_count = {}
# For each transaction
for i, row in data.iterrows(): 
    transaction = set()
    # For each product in the transaction
    for product in row: 
        # If valid product
        if type(product) is str: 
            transaction.add(product)
            if product in products_count: 
                products_count[product] += 1
            else:
                products_count[product] = 1
    transactions.append(transaction)        

# Calculate support
total = len(data)
products_support = { k : v / total for k, v in products_count.items() if v / total > MIN_SUPPORT }

# Count pairs of products
prod_pair_count = {}
# Iterate through products that have a support higher then treshold
for main_product in products_support.keys(): 
    # Iterate through the transactions
    for transaction in transactions: 
        if main_product in transaction: 
            for product in transaction: 
                if product in products_support and product != main_product:
                    product_pair = main_product, product
                    if product_pair in prod_pair_count:
                        prod_pair_count[product_pair] += 1
                    else:
                        prod_pair_count[product_pair] = 1


# Calculate confidence & lift
products_confidence = {}
lifts = []
for product_pair, count in prod_pair_count.items():
    product_1, product_2 = product_pair
    confidence = count / products_count[product_2]
    if confidence > MIN_CONFIDENCE:
        lift = confidence / products_support[product_1]    
        if lift > MIN_LIFT:
            lifts.append((product_pair, lift))

# Sort by lift
lifts.sort(key = lambda lift_tuple: lift_tuple[1], reverse = True) 

print('Results:')
print(*lifts, sep = '\n')