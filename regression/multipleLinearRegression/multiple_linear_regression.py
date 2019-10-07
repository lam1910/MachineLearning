#importing library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv(r"~/machineLearning/Machine-Learning-A-Z-New/Machine Learning A-Z New/Part 2 - Regression/Section 5 - Multiple Linear Regression/50_Startups.csv")
#create independent matrix of X (all rows, all column except the last which is the result or y)
X = dataset.iloc[:, :-1].values
#create dependent variable vector y (0-based), 4 columns
y = dataset.iloc[:, 4].values

#need encoding
#copy to data_preprocessing_template.py to runs
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelEncoder_X = LabelEncoder()
X[:, 3] = labelEncoder_X.fit_transform(X[:, 3])
#solution dummy column: instead of one column now it has n column (with n = number of different values (categories) in the original column
oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = oneHotEncoder.fit_transform(X).toarray()


#dummy variable created: index[0] for california, index[1] for florida, index[2] for new york
#avoid the Dummy variable trap (will be automated in python)(but do it anyway learning purpose) (remove california)
X = X[:, 1:]

#overfitting the dataset
#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)    #random_state for consistency

#need to scale if features have different range
#feature scalling no need because library will do it
'''from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

#dummy variable scalling depend on context'''

#mutiple linear regression
#fiting multiple linear regression in the trainning set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#predicting the Test set results
y_pred = regressor.predict(X_test)                                                              #vector prediction of the model


#no need to visualisation because of multiple dimension problem

#backward elimination
import statsmodels.formula.api as sm
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)
X_opt = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()                                           #step 1, 2 done
regressor_OLS.summary()
X_opt = X[:, [0, 1, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()                                                                         #step 3, 4, 5
X_opt = X[:, [0, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()                                                                         #step 3, 4, 5
X_opt = X[:, [0, 3, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()                                                                         #step 3, 4, 5
X_opt = X[:, [0, 3]]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()                                                                         #step 3, 4, 5
#backward elemination give only 1 element

