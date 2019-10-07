#importing library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv(r"~/machineLearning/Machine-Learning-A-Z-New/Machine Learning A-Z New/Part 6 - Reinforcement Learning/Section 33 - Thompson Sampling/Ads_CTR_Optimisation.csv")

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

#implement thompson sampling
number_of_reward_1 = [0] * d
number_of_reward_0 = [0] * d
ads_selected = []
total_reward_thompson = 0
for n in range(0, N):
    ad = 0
    max_random = 0
    for i in range(0, d):
        random_beta = random.betavariate(number_of_reward_1[i] + 1, number_of_reward_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[n, ad]
    if reward == 1:
        number_of_reward_1[ad] += 1
    else:
        number_of_reward_0[ad] += 1
    total_reward_thompson += reward

#visualizing the result histogram ucb
g = plt.figure(2)
plt.hist(ads_selected)
plt.title("Histogram of ads selection")
plt.xlabel("Ads")
plt.ylabel("Number of times each ad is selected")
plt.show()
