#importing library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv(r"~/machineLearning/Machine-Learning-A-Z-New/Machine Learning A-Z New/Part 4 - Clustering/Section 25 - Hierarchical Clustering/Mall_Customers.csv")
#create independent matrix of X (all rows, all column except the last which is the result or y)
X = dataset.iloc[:, [3, 4]].values

#using the dendrogram to pick the number of clusters
#using scipy
import scipy.cluster.hierarchy as sch
dendrogram = sch.dendrogram(sch.linkage(X, method = 'ward'))
f = plt.figure(1)
plt.title('Dendrogram')
plt.xlabel('Customers')
plt.ylabel('Euclidean distance')

#hierachical clusterings
#fitting hierachical clustering to dataset
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = 'euclidean', linkage = 'ward')
y_hc = hc.fit_predict(X)

#visualizing the clusters
g = plt.figure(2)
plt.scatter(X[y_hc == 0, 0], X[y_hc == 0, 1], c = 'red', label = 'Careful')
plt.scatter(X[y_hc == 1, 0], X[y_hc == 1, 1], c = 'blue', label = 'Standard')
plt.scatter(X[y_hc == 2, 0], X[y_hc == 2, 1], c = 'green', label = 'Target')
plt.scatter(X[y_hc == 3, 0], X[y_hc == 3, 1], c = 'cyan', label = 'Careless')
plt.scatter(X[y_hc == 4, 0], X[y_hc == 4, 1], c = 'magenta', label = 'Sensible')
plt.title("Clusters of clients")
plt.xlabel('Annual income (k$)')
plt.ylabel('Spending score (1-100)')
plt.legend()
plt.show()