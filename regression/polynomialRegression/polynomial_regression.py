#importing library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv(r"~/machineLearning/Machine-Learning-A-Z-New/Machine Learning A-Z New/Part 2 - Regression/Section 6 - Polynomial Regression/Position_Salaries.csv")
#create independent matrix of X (all rows, all column except the last which is the result or y)
X = dataset.iloc[:, 1: 2].values
#create dependent variable vector y (0-based), 4 columns
y = dataset.iloc[:, 2].values

#overfitting the dataset
#splitting the dataset into training set and test set (no need due to the amount of observation)
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)    #random_state for consistency"""

#need to scale if features have different range
#feature scalling
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#dummy variable scalling depend on context'''

#linear regression for comparision
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)


#polynomial regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 4)
X_poly = poly_reg.fit_transform(X)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)


#visuallising the tranning set results in linear regresion
plt.scatter(X, y, color = "red")
plt.plot(X, lin_reg.predict(X), color = "green")
plt.title('Salary via position (Linear regression)')
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()
#visuallising the tranning set results in polynomial regresion
X_grid = np.arange(min(X), max(X) + 0.1, 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = "red")
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = "green")
plt.title('Salary via position (Polynomial regression)')
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()

#predict with linear regression
print(lin_reg.predict([[6.5]])[0])
#predict with polynomial regression
print(lin_reg_2.predict(poly_reg.fit_transform([[6.5]]))[0])