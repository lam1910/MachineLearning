import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.formula.api as sm
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR



dataset = pd.read_excel(io = r"~/VSCode/Python/machinelearning/exercise/processed_sale_data.xlsx", sheet_name = 0
                        , usecols = [1, 2, 3, 4, 5, 6]
                        , dtype = {'storeCode': str, 'productCode': str, 'productType': str})


"""
#put different dataframe for different productCode

prod_code = dataset['productCode'].unique().tolist()
prods = []
for i in range(len(prod_code)):
    prods.append(dataset.loc[dataset.productCode == prod_code[i]])

Xs = []
ys = []
labelEncoder = LabelEncoder()
oneHotEncoder = OneHotEncoder(categorical_features = [1, 3])
for i in range(len(prods)):
    X = prods[i].iloc[:, [0, 1, 3, 4]].values
    y = prods[i].iloc[:, 5].values
    X[:, 0] = labelEncoder.fit_transform(X[:, 0])
    X = oneHotEncoder.fit_transform(X).toarray()
    Xs.append(X)
    ys.append(y)

#productCode as searching criteria
#treat 2 column storeCode and productType as categorical_data
#try for product 1 (productCode = '457401')
prod_0 = Xs[0]
result_0 = ys[0]


X_train, X_test, y_train, y_test = train_test_split(prod_0, result_0, test_size = 0.8, random_state = 0)    #random_state for consistency

#Adj_R_squared is low appox. 0.395 for multiple linear regression for the first item
#multiple linear regression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)


prod_0 = np.append(arr = np.ones((431, 1)).astype(int), values = prod_0, axis = 1)
X_opt = prod_0[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 22, 23, 24, 25, 26, 27, 28]]               #adj R-squared = 0.395 (best-value)
regressor_OLS = sm.OLS(endog = result_0, exog = X_opt).fit()                                              #step 1, 2 done

X_opt_train, X_opt_test, y_opt_train, y_opt_test = train_test_split(X_opt, result_0, test_size = 0.8, random_state = 0)
y_pred = regressor_OLS.predict(X_opt_test)


#decision tree demo first item negative r squared
regressor_t = DecisionTreeRegressor(criterion = 'friedman_mse', random_state = 0, max_features = 'sqrt', presort = True)
regressor_t.fit(X_train, y_train)

#predict with decision tree
#X_test = X_test.reshape(len(X_test), 1)
y_pred = regressor_t.predict(X_test)
metrics.mean_squared_error(y_test, y_pred)
metrics.max_error(y_test, y_pred)
metrics.mean_absolute_error(y_test, y_pred)
metrics.explained_variance_score(y_test, y_pred)
metrics.r2_score(y_test, y_pred)

#random forest for first item variance_score = 0.32 if round down
regressor_r = RandomForestRegressor(n_estimators = 300, criterion = 'mse', random_state = 0, max_features = 'sqrt', warm_start = True)
regressor_r.fit(X_train, y_train)

y_pred = regressor_r.predict(X_test)
y_pred_round_down = []
for iny_pred in y_pred:
    y_pred_round_down.append(int(iny_pred))

y_pred_round_up = []
for iny_pred in y_pred_round_down:
    y_pred_round_down.append(iny_pred + 1)
"""
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 5].values
labelEncoder = LabelEncoder()
X[:, 0] = labelEncoder.fit_transform(X[:, 0])

X_seg = X[:35243, :]
y_seg = y[:35243]
oneHotEncoder = OneHotEncoder(categorical_features = [1, 2, 3])
X_seg = oneHotEncoder.fit_transform(X_seg).toarray()

X_train = X_seg[:25285, :]
y_train = y_seg[:25285]
X_test = X_seg[25285:, :]
y_test = y_seg[25285:]


#multiple linear regression for all features week 3 adj. R-squared = 0.487
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred_l = regressor.predict(X_test)
metrics.r2_score(y_test, y_pred_l)
#
metrics.mean_absolute_error(y_test, y_pred_l)
#
metrics.mean_squared_error(y_test, y_pred_l)
#


def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((35243, 1539)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y_seg, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:, j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y_seg, x).fit()
                    adjR_after = tmp_regressor.rsquared_adj.astype(float)
                    if (adjR_before >= adjR_after):
                        x_rollback = np.hstack((x, temp[:, [0, j]]))
                        x_rollback = np.delete(x_rollback, j, 1)
                        print(regressor_OLS.summary())
                        return x_rollback
                    else:
                        continue
    regressor_OLS.summary()
    return x


X_seg_opt = np.append(arr = np.ones((35243, 1)).astype(int), values = X_seg, axis = 1)
X_feat = list(range(1539))
SL = 0.05
X_opt = X_seg_opt[:, X_feat]
X_Modeled = backwardElimination(X_opt, SL)
regressor.fit(X_Modeled, y_train)



#random forest for all features r2 score = 0.80 week3
regressor_r = RandomForestRegressor(n_estimators = 300, criterion = 'mse', random_state = 0, max_features = 'sqrt'
                                    , warm_start = True)
regressor_r.fit(X_train, y_train)
y_pred_r = regressor_r.predict(X_test)
metrics.r2_score(y_test, y_pred_r)
#0.7993773092470942
metrics.mean_absolute_error(y_test, y_pred_r)
#6.870585234986945
metrics.explained_variance_score(y_test, y_pred_r)
#0.7993977604664907
metrics.max_error(y_test, y_pred_r)
#7562.889999999999
metrics.mean_squared_error(y_test, y_pred_r)
#

#for week 6 (more features -> retrain) r2 score = 0.83
X_seg_1 = X[:72527, :]
y_seg_1 = y[:72527]
X_seg_1 = oneHotEncoder.fit_transform(X_seg_1).toarray()
X_train_1 = X_seg_1[:60293, :]
y_train_1 = y_seg_1[:60293]
X_test_1 = X_seg_1[60293:, :]
y_test_1 = y_seg_1[60293:]
regressor_r = RandomForestRegressor(n_estimators = 300, criterion = 'mse', random_state = 0, max_features = 'sqrt'
                                    , warm_start = False)
regressor_r.fit(X_train_1, y_train_1)
y_pred_r_1 = regressor_r.predict(X_test_1)
metrics.r2_score(y_test_1, y_pred_r_1)
#0.8359427240408955
metrics.mean_absolute_error(y_test_1, y_pred_r_1)
#7.967871767478611
metrics.explained_variance_score(y_test_1, y_pred_r_1)
#0.8360260911675206
metrics.max_error(y_test_1, y_pred_r_1)
#5796.9366666666665
metrics.mean_squared_error(y_test, y_pred_r_1)
#

#SVR for week 3
sc_X = StandardScaler()
sc_y = StandardScaler()
X_seg_1 = X[:72527, :]
y_seg_1 = y[:72527]
X_seg_1 = oneHotEncoder.fit_transform(X_seg_1).toarray()
X_seg_1 = sc_X.fit_transform(X_seg_1)
y_seg_1 = sc_y.fit_transform(y_seg_1)

X_train_1 = X_seg_1[:60293, :]
y_train_1 = y_seg_1[:60293]
X_test_1 = X_seg_1[60293:, :]
y_test_1 = y_seg_1[60293:]

regressor_svr = SVR(kernel = 'rbf')
regressor_svr.fit(X_train_1, y_train_1)
y_pred_svr = sc_y.inverse_transform(regressor_svr.predict(X_test_1))





