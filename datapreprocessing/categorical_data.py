import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv(r"~/machineLearning/Machine-Learning-A-Z-New/Machine Learning A-Z New/Part 1 - Data Preprocessing//Data.csv")
#create independent matrix of X (all rows, all column except the last which is the result or y)
X = dataset.iloc[:, :-1].values
#create dependent variable vector y (0-based), 4 columns
y = dataset.iloc[:, 3].values
#categorical variable is a variable that can take on one of a limited, and usually fixed number of possible values, assigning each
# individual or other unit of observation to a particular group or nominal category on the basis of some qualitative property.

#need encoding
#in tutorial encode country and purchased
#copy to data_preprocessing_template.py to runs
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 0] = labelEncoder_X.fit_transform(X[:, 0])                                  #change for country be careful with comparision because the original data cannot be compare
#solution dummy column: instead of one column now it has n column (with n = number of different values (categories) in the original column
oneHotEncoder = OneHotEncoder(categorical_features = [0])
X = oneHotEncoder.fit_transform(X).toarray()
labelEncoder_y = LabelEncoder()
y = labelEncoder_y.fit_transform(y)