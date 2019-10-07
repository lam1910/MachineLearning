
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import gc
import sklearn

from sklearn.model_selection import GridSearchCV

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import xgboost

from sklearn.metrics import r2_score

def smape(A, F):
    return 100/len(A) * np.sum(np.abs(F - A) / (np.abs(A) + np.abs(F) + np.finfo(float).eps))


scaler = StandardScaler()
lbe    = LabelEncoder()

#===============READ THE ALREADY PROCESSED DF ==========================
"""
df_final_table = pd.read_excel('all_sale_20week_2607.xlsx')
#df_final_table = pd.read_excel('all_sale124.xlsx')
df_final_table = df_final_table.iloc[:, 1:]
"""

#data_store = pd.HDFStore('allsale_2907_month.h5')
data_store = pd.HDFStore('aha_20190805.h5')
# Retrieve data using key
df_final_table = data_store['df_final_table']
data_store.close()


df_final_table = df_final_table[['date',"month", "quarter", "year", 'storeCode', 'productCode', 'productType', 'price', "valentine", "new_year", "mid_autumn",
       'saleLast1Month', 'saleLast2Month', 'saleLast3Month', 'saleLast4Month', 'saleLast5Month', 'saleLast6Month', 'saleLast12Month',
       "growth1", "growth2", "growth3", "growth4", "growth5", "growth6", "growth12",
       "max1Month", "min1Month", "mean1Month", "mad1Month", "std1Month", "max2Month", "min2Month", "mean2Month", "mad2Month", "std2Month", "max4Month", "min4Month", "mean4Month", "mad4Month", "std4Month", "max5Month", "min5Month", "mean5Month", "mad5Month", "std5Month",
       "max3Month", "min3Month", "mean3Month", "mad3Month", "std3Month",  "max6Month", "min6Month", "mean6Month" , "mad6Month", "std6Month", "max12Month", "min12Month", "mean12Month", "mad12Month", "std12Month",
       'itemCount']]

df_final_table[['storeCode', 'productCode', 'productType']] = df_final_table[['storeCode', 'productCode', 'productType']].astype(str)
lbe = lbe.fit(df_final_table[['storeCode', 'productCode', 'productType']].values.ravel())
#=======================================================================
df_final_table.replace([np.inf, -np.inf], 1000000, inplace = True)
df_final_table.sort_values(by = ["date", "productCode", "storeCode"], ascending = [True, True, True], inplace = True)


PRED_DATE = int((2019 - 2014)*12 + 6 - 12 )

df_verification = df_final_table[df_final_table["date"] == PRED_DATE]
df_final_table = df_final_table[df_final_table["date"] < PRED_DATE]

# CUT CUT FUCKING CUT
"""
min_date = df_final_table["date"].min() + 8
df_final_table = df_final_table[df_final_table["date"] >= min_date]
df_final_table.reset_index(inplace = True, drop = True)
"""

start_time = time.time()
no_of_week_test = 3
min_date = df_final_table["date"].max() - no_of_week_test
split_index = df_final_table.index[df_final_table['date'] == min_date +1].tolist()[0]


"""
df_final_table["saleLast2Week"] = df_final_table["saleLast2Week"].apply(lambda x: x/2)
df_final_table["saleLast4Week"] = df_final_table["saleLast4Week"].apply(lambda x: x/4)

df_final_table["categorySaleLocal2"] = df_final_table["categorySaleLocal2"].apply(lambda x: x/2)
df_final_table["categorySaleLocal4"] = df_final_table["categorySaleLocal4"].apply(lambda x: x/4)

df_final_table["categorySaleGlobal2"] = df_final_table["categorySaleGlobal2"].apply(lambda x: x/2)
df_final_table["categorySaleGlobal4"] = df_final_table["categorySaleGlobal4"].apply(lambda x: x/4)
"""


columns_to_encode = ['productCode', 'productType', 'storeCode']

#df_final_table[columns_to_encode] = df_final_table[columns_to_encode].astype(str)
columns_to_scale  = ['date',"month", "year", 'price',# "valentine", "new_year", "mid_autumn",
       'saleLast1Month', 'saleLast2Month', 'saleLast3Month', 'saleLast4Month', 'saleLast5Month','saleLast12Month',
       "growth1", "growth2", "growth3", "growth4", "growth5","growth12",
       "max2Month", "min2Month", "mean2Month", "mad2Month", "std2Month", "max4Month", "min4Month", "mean4Month", "mad4Month", "std4Month", "max5Month", "min5Month", "mean5Month", "mad5Month", "std5Month",
       "max3Month", "min3Month", "mean3Month", "mad3Month", "std3Month", "max12Month", "min12Month", "mean12Month", "mad12Month", "std12Month"]
       
answer_column = ['itemCount']
# Instantiate encoder/scaler


# Scale and Encode Separate Columns
#scaled_columns  = scaler.fit_transform(df_final_table[columns_to_scale]) 
#scaled_columns_2 = scaler.fit_transform(df_final_table[answer_column])

scaled_columns  = df_final_table[columns_to_scale]
scaled_columns_2 = df_final_table[answer_column]

df_encoded_columns = df_final_table[columns_to_encode]

encoded_columns = df_encoded_columns.apply(lbe.transform)
#encoded_columns =    ohe.fit_transform(df_final_table[columns_to_encode])

# Concatenate (Column-Bind) Processed Columns Back Together
processed_data = np.concatenate([scaled_columns, encoded_columns], axis=1)

#del df_final_table
last_month = df_final_table["saleLast1Month"]


X = processed_data
y = scaled_columns_2["itemCount"].ravel()

X_train = X[0: split_index, :]
y_train = y[0: split_index] 

X_test = X[split_index: -1, :]
y_test = y[split_index: -1]

last_month_test = last_month[split_index: -1]

print('Starting training...')
# train
eval_set = [(X_train, y_train), (X_test, y_test)]
eval_metric = ["mae",  "rmse"]

xgb_model = xgboost.XGBRegressor(silent=False,
                      learning_rate=0.025,
                      colsample_bytree = 0.8,
                      subsample = 0.8,
                      objective='reg:squarederror', 
                      n_estimators=2000,
                      monotone_constraints = (0,0,0,0, 1, 1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0 ),
                      max_depth=8)
#xgb_model = xgboost.XGBRegressor(objective="reg:squarederror", eval_set = eval_set, eval_metric = eval_metric,  )

xgb_model.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set, verbose=True, early_stopping_rounds = 30)


"""
params = {
        "learning_rate": [0.002, 0.01, 0.02, 0.025, 0.05, 0.1],
        "colsample_bytree" : [1, 0.9],
        "subsample" : [1, 0.9],
        "max_depth": [8]
        }

# Initialize XGB and GridSearch
xgb = xgboost.XGBRegressor(objective='reg:squarederror',
                      n_estimators=2000, 
                      reg_alpha = 0.3,
                      verbosity =0) 

grid = GridSearchCV(xgb, params)
grid.fit(X_train, y_train, eval_metric=eval_metric, eval_set=eval_set, early_stopping_rounds = 30  )

# Print the r2 score
print("BEST SCORE ", r2_score(y_test, grid.best_estimator_.predict(X_test))) 
"""

# Save the file
y_pred = xgb_model.predict(X_test)

training_time = time.time() - start_time 
print ("total training time ", training_time , "sec")

print('Starting predicting...')
# predict
#y_pred = xgb_model.predict(X_test)
# eval

#true_y_test = scaler.inverse_transform(y_test)
#true_y_pred = scaler.inverse_transform(y_pred)
true_y_test = y_test
true_y_pred = y_pred

x1 = np.arange(0, len(y_test), 1)
plt.scatter(x1, true_y_test, color = 'red')
plt.scatter(x1, true_y_pred, color = 'blue')
plt.show()



for index, value in enumerate(true_y_pred):
    if value < 1:
        true_y_pred[index] = 0
true_y_pred = np.round(true_y_pred) 
print("r2 ",sklearn.metrics.r2_score(true_y_test, true_y_pred))       
print("Mean absolute error ", sklearn.metrics.mean_absolute_error(true_y_test, true_y_pred))
print("Mean squared error ", sklearn.metrics.mean_squared_error(true_y_test, true_y_pred))
print("SMAPE ", smape(true_y_test, true_y_pred))

print("average y: ", np.mean(true_y_test))
print (len(true_y_pred))

#print("r2 lw ",sklearn.metrics.r2_score(true_y_test, last_month_test))       
#print("Mean absolute error lw", sklearn.metrics.mean_absolute_error(true_y_test, last_month_test))
#print("Mean squared error lw", sklearn.metrics.mean_squared_error(true_y_test, last_month_test))
#print("SMAPE lw", smape(true_y_test, last_month_test))



# feature importance
print(xgb_model.feature_importances_)
# plot
plt.bar(range(len(xgb_model.feature_importances_)), xgb_model.feature_importances_)
plt.show()

diff = np.subtract(true_y_test, true_y_pred)
diff = abs(diff)
plt.plot(diff)
plt.show()

#calculate necessary parameters for new prediction

#if the date is > max date, create df_verification manually

df_verification.reset_index(drop = True, inplace = True)
#check new prediction
# Scale and Encode Separate Columns
#scaled_verify_columns  = scaler.transform(df_verification[columns_to_scale]) 
scaled_verify_columns  = df_verification[columns_to_scale]
df_encoded_verify_columns = df_verification[columns_to_encode]
encoded_verify_columns = df_encoded_verify_columns.apply(lbe.transform)
#encoded_columns =    ohe.fit_transform(df_final_table[columns_to_encode])

# Concatenate (Column-Bind) Processed Columns Back Together
processed_verify_data = np.concatenate([scaled_verify_columns, encoded_verify_columns], axis=1)

X_verify = processed_verify_data

y_verify = xgb_model.predict(X_verify)

for index, value in enumerate(y_verify):
    if value < 1:
        y_verify[index] = 0
    
y_verify = np.round(y_verify) 
