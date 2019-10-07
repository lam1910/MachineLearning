#importing library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv(r"~/machineLearning/Machine-Learning-A-Z-New/Machine Learning A-Z New/Part 2 - Regression/Section 7 - Support Vector Regression (SVR)/Position_Salaries.csv")
#create independent matrix of X (all rows, all column except the last which is the result or y)
X = dataset.iloc[:, 1: 2].values
#create dependent variable vector y (0-based), 4 columns
y = dataset.iloc[:, 2: 3].values

#overfitting the dataset
#splitting the dataset into training set and test set (if many observation remove comment)
"""from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)    #random_state for consistency"""

#need to scale if features have different range
#feature scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
sc_y = StandardScaler()
X = sc_X.fit_transform(X)
y = sc_y.fit_transform(y)

#dummy variable scalling depend on context

#support vector regression
#fiting svr model to dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X, y)


#predict with svr
y_pred = sc_y.inverse_transform(regressor.predict(sc_X.transform(np.array([[6.5]]))))

print(y_pred[0])

#visuallising the tranning set results in svr model
X_grid = np.arange(min(X), max(X) + 0.1, 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regressor.predict(X_grid), color = "green")
plt.title('Salary via position (Regression model)')
plt.xlabel("Position level")
plt.ylabel("Salary")
plt.show()