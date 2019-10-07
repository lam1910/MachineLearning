#importing library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv(r"~/machineLearning/Machine-Learning-A-Z-New/Machine Learning A-Z New/Part 6 - Reinforcement Learning/Section 32 - Upper Confidence Bound (UCB)/Ads_CTR_Optimisation.csv")

#implementing random selection for comparision
import random
N = 10000
d = 10
ads_selected = []
total_reward_radom = 0
for n in range(0, N):
    ad = random.randrange(d)
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    total_reward_radom += reward

#visualizing the result histogram
f = plt.figure(1)
plt.hist(ads_selected)
plt.title("Histogram of ads selection")
plt.xlabel("Ads")
plt.ylabel("Number of times each ad is selected")

#implement UCB
import math
number_of_selections = [0] * d
sum_of_reward = [0] * d
ads_selected = []
total_reward_ucb = 0
for n in range(0, N):
    ad = 0
    max_upper_bound = 0
    for i in range(0, d):
        if number_of_selections[i] > 0:
            average_reward = sum_of_reward[i] / number_of_selections[i]
            delta_i = math.sqrt(3/2 * math.log(n + 1) / number_of_selections[i])            #array start from 0
            upper_bound = average_reward + delta_i
        else:
            upper_bound = 1e400
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_of_selections[ad] += 1
    reward = dataset.values[n, ad]
    sum_of_reward[ad] += reward
    total_reward_ucb += reward

#visualizing the result histogram ucb
g = plt.figure(2)
plt.hist(ads_selected)
plt.title("Histogram of ads selection")
plt.xlabel("Ads")
plt.ylabel("Number of times each ad is selected")
plt.show()
