#data preproccessing
#importing library
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the dataset
dataset = pd.read_csv(r"~/machineLearning/Machine-Learning-A-Z-New/Machine Learning A-Z New/Part 2 - Regression/Section 4 - Simple Linear Regression/Salary_Data.csv")
#create independent matrix of X (all rows, all column except the last which is the result or y)
X = dataset.iloc[:, :-1].values
#create dependent variable vector y (0-based), 4 columns
y = dataset.iloc[:, 1].values

#overfitting the dataset
#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)    #random_state for consistency

#need to scale if features have different range
#feature scalling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
y = sc_y.fit_transform(y)


#dummy variable scalling depend on context'''
#in this simple linear regression, no need to scale

#simple linear regression
#fiting simple linear regression in the trainning set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the Test set results
y_pred = regressor.predict(X_test)                                                              #vector prediction of the model

#visuallising the tranning set results
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "green")
plt.title('Salary via experience (Training set)')
plt.xlabel("Year of experience")
plt.ylabel("Salary")
plt.show()

#visuallising the test set results
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, regressor.predict(X_train), color = "green")
plt.title('Salary via experience (Test set)')
plt.xlabel("Year of experience")
plt.ylabel("Salary")
plt.show()
