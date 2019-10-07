import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv(r"~/machineLearning/Machine-Learning-A-Z-New/Machine Learning A-Z New/Part 1 - Data Preprocessing//Data.csv")
#create independent matrix of X (all rows, all column except the last which is the result or y)
X = dataset.iloc[:, :-1].values
#create dependent variable vector y (0-based), 4 columns
y = dataset.iloc[:, 3].values

#taking care of missing data
#copy to data_preprocessing_template.py to run
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = np.nan, strategy = 'mean', axis = 0)         #no need strategy = 'mean' because of its default status
imputer = imputer.fit(X[:, 1:3])                                                #missing data on column no 2, 3 (check for all dataset)
X[:, 1:3] = imputer.transform(X[:, 1:3])