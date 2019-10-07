#importing library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# no need to do anything, just do it straight in dataset
from associationrulelearning.apriori import apyori

dataset = pd.read_csv(r"~/machineLearning/Machine-Learning-A-Z-New/Machine Learning A-Z New/Part 5 - Association Rule Learning/Section 28 - Apriori/Market_Basket_Optimisation.csv", header=None)
# ceating a list to feed to algorithm
transactions = []
for i in range(0, 7501):
    transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])

from associationrulelearning.apriori.apyori import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)

#visualize the result
results = list(rules)