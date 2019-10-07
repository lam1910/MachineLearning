# Importing the libraries
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
#import statsmodels.formula.api as sm
import sklearn.metrics as metrics
#from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
#from sklearn.preprocessing import StandardScaler
#from sklearn.svm import SVR


print("________________________________")
og_dataset = pd.read_excel('MASS_corele_dulieu_20190919.xlsx', header = None)
og_dataset.columns = ["date", "productCode", "productType", "storeCode", "price", "itemCount"]

dataset = og_dataset.iloc[:, :]
dataset.reset_index(inplace = True)
dataset.drop(columns = "index", axis = 1, inplace = True)
dataset.iloc[:, 0] = pd.to_datetime(dataset.iloc[:, 0], format = "%Y%m%d" )

dataset = dataset.fillna(value = 'default', axis = 1)
dataset = dataset.astype({'productCode': 'str', 'productType': 'str', 'storeCode': 'str'})

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 5].values
labelEncoder = LabelEncoder()
X[:, 0] = labelEncoder.fit_transform(X[:, 0])
X[:, 1] = labelEncoder.fit_transform(X[:, 1])
X[:, 2] = labelEncoder.fit_transform(X[:, 2])
X[:, 3] = labelEncoder.fit_transform(X[:, 3])


oneHotEncoder = OneHotEncoder(categorical_features = [1, 2, 3])
X = oneHotEncoder.fit_transform(X).toarray()


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)


# regressor = LinearRegression()
# regressor.fit(X_train, y_train)
# y_pred_l = regressor.predict(X_test)
# print(metrics.r2_score(y_test, y_pred_l))
# #print(
# print(metrics.mean_absolute_error(y_test, y_pred_l))
# #
# print(metrics.mean_squared_error(y_test, y_pred_l))

regressor_r = RandomForestRegressor(n_estimators = 300, criterion = 'mse', random_state = 0, max_features = 'sqrt'
                                    , warm_start = True)
regressor_r.fit(X_train, y_train)
y_pred_r = regressor_r.predict(X_test)
print(metrics.r2_score(y_test, y_pred_r))
#
print(metrics.mean_absolute_error(y_test, y_pred_r))
#
print(metrics.explained_variance_score(y_test, y_pred_r))
#
print(metrics.max_error(y_test, y_pred_r))
#
print(metrics.mean_squared_error(y_test, y_pred_r))