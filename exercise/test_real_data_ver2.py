import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import statsmodels.formula.api as sm
import sklearn.metrics as metrics
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import ShuffleSplit
import collections


dataset = pd.read_csv(filepath_or_buffer = r"~/VSCode/Python/machinelearning/exercise/processed_sale_data_ver2.csv"
                        , usecols = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
                        , dtype = {'storeCode': str, 'productCode': str, 'productType': str})

X = dataset.iloc[:,[0, 1, 2, 3, 4, 6, 7, 8, 9, 10, 11]].values
y = dataset.iloc[:, 5].values
labelEncoder = LabelEncoder()
X[:, 0] = labelEncoder.fit_transform(X[:, 0])

X_seg = X[:36604, :]
y_seg = y[:36604]
oneHotEncoder = OneHotEncoder(categorical_features = [1, 2, 3])
X_seg = oneHotEncoder.fit_transform(X_seg).toarray()

X_train = X_seg[:25285, :]
y_train = y_seg[:25285]
X_test = X_seg[25285:, :]
y_test = y_seg[25285:]

#multiple linear regression for all features week 3 adj. R-squared = 0.710
regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred_l = regressor.predict(X_test)
metrics.r2_score(y_test, y_pred_l)
#0.43211571808238847
metrics.mean_absolute_error(y_test, y_pred_l)
#12.328244465220747
metrics.mean_squared_error(y_test, y_pred_l)
#36548.24160613644

def backwardElimination(x, SL):
    numVars = len(x[0])
    temp = np.zeros((25285, 1569)).astype(int)
    for i in range(0, numVars):
        regressor_OLS = sm.OLS(y_train, x).fit()
        maxVar = max(regressor_OLS.pvalues).astype(float)
        adjR_before = regressor_OLS.rsquared_adj.astype(float)
        if maxVar > SL:
            for j in range(0, numVars - i):
                if (regressor_OLS.pvalues[j].astype(float) == maxVar):
                    temp[:, j] = x[:, j]
                    x = np.delete(x, j, 1)
                    tmp_regressor = sm.OLS(y_train, x).fit()
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


X_seg_opt = np.append(arr = np.ones((25285, 1)).astype(int), values = X_train, axis = 1)
X_feat = list(range(1569))
SL = 0.05
X_opt = X_seg_opt[:, X_feat]
X_Modeled = backwardElimination(X_opt, SL)
regressor.fit(X_Modeled, y_train)

#random forest for all features r2 score = 0.812 week3
regressor_r = RandomForestRegressor(n_estimators = 300, criterion = 'mse', random_state = 0, max_features = 'sqrt'
                                    , warm_start = True)
regressor_r.fit(X_train, y_train)
y_pred_r = regressor_r.predict(X_test)
metrics.r2_score(y_test, y_pred_r)
#0.8116881381420423
metrics.mean_absolute_error(y_test, y_pred_r)
#6.567237995994935
metrics.explained_variance_score(y_test, y_pred_r)
#0.8116907253433028
metrics.max_error(y_test, y_pred_r)
#7166.893333333333
metrics.mean_squared_error(y_test, y_pred_r)
#12119.489205169679

#get columns with feature_importences_ >= 0.005
names = list(range(1570))
mapping = dict(zip(names, map(lambda x: round(x, 4), regressor_r.feature_importances_)))
colToPick = []
for key in mapping.keys():
    if mapping[key] >= 0.005:
        colToPick.append(key)


regressor_r_opt = RandomForestRegressor(n_estimators = 300, criterion = 'mse', random_state = 0)
X_r_opt = X_seg[:, colToPick]
regressor_r_opt.fit(X_r_opt[:25285, :], y_train)
y_pred_r_opt = regressor_r_opt.predict(X_r_opt[25285:, :])
metrics.r2_score(y_test, y_pred_r_opt)
#0.816941404606213
metrics.mean_absolute_error(y_test, y_pred_r_opt)
#6.611776673588489
metrics.explained_variance_score(y_test, y_pred_r_opt)
#0.8169540934518129
metrics.max_error(y_test, y_pred_r_opt)
#7068.083333333334
metrics.mean_squared_error(y_test, y_pred_r_opt)
#11781.396290701976

scores = collections.defaultdict(list)

#crossvalidate the scores on a number of different random splits of the data (not completed)
for train_idx, test_idx in ShuffleSplit(n_splits = 100, test_size = .3).split(X_seg, y_seg):
    X_r_opt_train, X_r_opt_test = X_seg[train_idx], X_seg[test_idx]
    y_r_opt_train, y_r_opt_test = y_seg[train_idx], y_seg[test_idx]
    r = regressor_r_opt.fit(X_r_opt_train, y_r_opt_train)
    acc = metrics.r2_score(y_r_opt_test, regressor_r_opt.predict(X_r_opt_test))
    for i in range(X.shape[1]):
        X_t = X_r_opt_test.copy()
        np.random.shuffle(X_t[:, i])
        shuff_acc = metrics.r2_score(y_r_opt_test, regressor_r_opt.predict(X_t))
        scores[names[i]].append((acc-shuff_acc)/acc)

print("Features sorted by their score:")
print(sorted([(round(np.mean(score), 4), feat) for
              feat, score in scores.items()], reverse=True))

#for week 6 (more features -> retrain) r2 score = 0.831
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
#0.8305485958635979
metrics.mean_absolute_error(y_test_1, y_pred_r_1)
#7.9168280102810025
metrics.explained_variance_score(y_test_1, y_pred_r_1)
#0.8306259440692015
metrics.max_error(y_test_1, y_pred_r_1)
#5820.65
metrics.mean_squared_error(y_test_1, y_pred_r_1)
#15047.114554120555
