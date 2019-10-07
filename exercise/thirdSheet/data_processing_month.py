# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import time
import gc
import sklearn
import lightgbm as lgb 


#PREDICT_DATE = datetime.datetime(year = 2019, month = 6, day = 30)
start_time = time.time() 
og_dataset = pd.read_excel('MASS_corele_dulieu_20190919.xlsx', header = None)
#og_dataset = pd.read_excel('thongke_aha24_20190805.xlsx', header = None)

og_dataset.columns = ["date", "productCode", "productType", "storeCode", "price", "itemCount"]

dataset = og_dataset.iloc[:, :]
dataset.reset_index(inplace = True)
dataset.drop(columns = "index", axis = 1, inplace = True) 
dataset.iloc[:, 0] = pd.to_datetime(dataset.iloc[:, 0], format = "%Y%m%d" )

dataset["revenue"] = dataset["price"] * dataset["itemCount"]

#columns_to_encode = ['productCode', 'productType', 'storeCode']

#dataset[columns_to_encode] = dataset[columns_to_encode].astype(str)

#dataset= dataset[dataset["productCode"]== 'VNL00325']
#dataset= dataset[dataset["storeCode"]== '88087' ]



"""
base_list = pd.read_excel('base_product.xlsx',index_col = 0)
base_list = base_list.iloc[:,2]

dataset= dataset[dataset.productCode.isin(base_list)]
"""
#re sample the data by week
dataset_month = dataset.groupby([pd.Grouper(key='date', freq='M'), pd.Grouper(key='productCode'), pd.Grouper(key='productType'), pd.Grouper(key='storeCode')]).agg({"price":"mean","itemCount":"sum", "revenue": "sum"})
dataset_month.reset_index(drop = False, inplace = True)

columns_to_encode = ['productCode', 'productType', 'storeCode']

dataset_month[columns_to_encode] = dataset_month[columns_to_encode].astype(str)

#dataset_month[["productCode","productType","storeCode"]] = dataset_month[["productCode","productType","storeCode"]].astype(str)


no_of_month_sampled = 37
max_date = dataset_month["date"].max()
min_date = max_date - datetime.timedelta(days = no_of_month_sampled * 30)
dataset_month = dataset_month[dataset_month["date"] >= min_date ] 
dataset_month = dataset_month[dataset_month["date"] <= max_date ]
dataset_month.reset_index(drop = True, inplace = True)


"""
k= 0
dropped_index = []
while k <= dataset_month.shape[0]-1:
    #print ("index = ", k," store ", dataset_month.iloc[k, 3], " year ", dataset_month.iloc[k,0].year)
    if (dataset_month.iloc[k, 3] == '14016')  and  (dataset_month.iloc[k,0].year < 2019):
        print ("index = ", k," store ", dataset_month.iloc[k, 3], " year ", dataset_month.iloc[k,0].year)
        #df_final_table.drop(df_final_table.index[k],axis = 0, inplace = True)
        dropped_index.append(k)
        #print ("df size after drop ", df_final_table.shape[0])
        #continue
    k = k+1

print ("total dropped rows ", len(dropped_index))
dataset_month.drop(dropped_index,axis = 0, inplace = True)
"""


#dataset_month= dataset_month[dataset_month["productCode"]== 'VNL00325']
"""
dataset_month= dataset_month[dataset_month["storeCode"]== '36040' ]
"""

sale_by_store = dataset_month.groupby(["storeCode"]).agg({"itemCount":"sum", "revenue": "sum"})
sale_by_product = dataset_month.groupby(["productCode"]).agg({"itemCount":"sum", "revenue": "sum"})
sale_by_type = dataset_month.groupby(["productType"]).agg({"itemCount":"sum", "revenue": "sum"})

max_date_2018 = datetime.datetime(year = 2019, month = 8, day = 1)
min_date_2018 = datetime.datetime(year = 2015, month = 12, day = 31)
total_sale_by_month = dataset_month#[dataset_month["productCode"]== 'VNL00325']
#total_sale_by_month = total_sale_by_month[total_sale_by_month["storeCode"]== '14017' ]
total_sale_by_month = total_sale_by_month.groupby(["date"]).agg({"itemCount":"sum", "revenue": "sum"})
total_sale_by_month.reset_index(inplace = True)
total_sale_by_month = total_sale_by_month[(total_sale_by_month["date"]> min_date_2018) ]
total_sale_by_month = total_sale_by_month[(total_sale_by_month["date"]< max_date_2018) ]


#REMOVE FUCKING BUTTER 
"""
df_product_sale = dataset_month.groupby(["productCode"]).agg({"itemCount":"sum"})
df_product_sale.sort_values(by = ["itemCount"], ascending = [False], inplace = True)
df_product_sale.reset_index(inplace = True)
median_sale = df_product_sale["itemCount"].median()

bad_product = []

for row_index, row in df_product_sale.iterrows():
    if df_product_sale.iloc[row_index,1] > median_sale * 1000:
        #df_product_sale.iloc[row_index,1] = float(df_product_sale.iloc[row_index,1] / 1000)
        #df_product_sale.iloc[row_index,1] = float(df_product_sale.iloc[row_index,1] / 1000)
        print ("SALE WAY TOO IRREGULAR ", df_product_sale.iloc[row_index,0], "total remove ", delete_count )
        bad_product.append(df_product_sale.iloc[row_index,0])


delete_count = 0
deleted_index = []

for row_index, row in dataset_month.iterrows():
    if dataset_month.iloc[row_index, 1] in bad_product:
        delete_count +=1
        dataset_month.iloc[row_index, 4] = dataset_month.iloc[row_index, 4] * 1000
        dataset_month.iloc[row_index, 5] = float(dataset_month.iloc[row_index, 5]) / 1000
        #print ("FUCKING BUTTER ", dataset_month.iloc[row_index,1], "total remove ", delete_count )
        print ("FUCKING BUTTER ", dataset_month.iloc[row_index,1], "total remove ", delete_count )
        print ("new price ", dataset_month.iloc[row_index, 4], " new sale ", float(dataset_month.iloc[row_index, 5]))
        deleted_index.append(row_index)
#dataset_month.drop(deleted_index,axis = 0, inplace = True)
#dataset_month.reset_index(drop = True, inplace = True)

"""

time1= time.time()
product_count = dataset_month.groupby(["productCode"]).agg({"productType":"first"})
product_count.reset_index(inplace = True)
#base_list = base_list.to_frame()
#base_list["total_count"] = base_list["productCode"].apply(lambda x: product_count[product_count["productCode"] == x]["itemCount"].sum())
store_count = dataset_month.groupby(["storeCode"]).agg({"itemCount":"sum"})
store_count.reset_index(inplace = True)
date_count = dataset_month.groupby(["date"]).agg({"itemCount":"count"})
date_count.reset_index(inplace = True)

# Make sure all store and products are present
df_final_table = pd.MultiIndex.from_product([date_count["date"], product_count["productCode"], store_count["storeCode"]], names=['date','productCode', 'storeCode']).to_frame()
dataset_month.set_index(["date", "productCode", "storeCode"], inplace = True)

df_final_table = df_final_table.join(dataset_month, how = 'left')
df_final_table.reset_index(drop = True, inplace = True)
time2 = time.time()
print ("product time ", time2 - time1)
#total_everything_2["productType"] = total_everything_2["productCode"].apply(lambda x: product_count[product_count["productCode"] == x].iloc[0,2])
df_final_table.drop("productType",axis = 1, inplace = True)
df_final_table = pd.merge(df_final_table, product_count, on='productCode', how='inner')
print ("merge time ", time.time() - time2)

df_final_table.sort_values(by = ["productCode", "storeCode", "date"], ascending = [True, True, True], inplace = True)
df_final_table.reset_index(inplace = True)

df_final_table["itemCount"] = df_final_table["itemCount"].fillna(0) 


#add more row for week with no sale
"""
df_price = dataset_month.set_index(
    ["storeCode", "productCode", "productType", "date"])[["price"]].unstack(
        level=-1)
df_price.columns = df_price.columns.get_level_values(1)
df_price.reset_index()
df_final_price = pd.pivot_table(df_price,columns = ["storeCode","productCode","productType"]).reset_index()

df_final_table = df_final_price
df_final_price.columns = ["date", "storeCode", "productCode", "productType", "price"]
del df_price, df_final_price
gc.collect()

df_count = dataset_month.set_index(
    ["storeCode", "productCode", "productType", "date"])[["itemCount"]].unstack(
        level=-1).fillna(0)
df_count.columns = df_count.columns.get_level_values(1)
df_count.reset_index()
df_final_count = pd.pivot_table(df_count,columns = ["storeCode","productCode","productType"]).reset_index()

df_final_table['itemCount'] = 0
df_final_table.iloc[:, -1] = df_final_count.iloc[:, -1]

del df_count,df_final_count
gc.collect()
"""


#---------------------------- CALCULATE PAST 1 2 3 AND 4 MONTH SALES -----------------------------------------------------
print("Adding past weeks sale as new columns: ")


df_final_table["month"] = df_final_table["date"].dt.month
df_final_table["quarter"] = df_final_table["date"].dt.quarter
df_final_table["year"] = df_final_table["date"].dt.year




#df_final_table.iloc[-1, 5] = 9999999999
"""
df_final_table['saleLast1Month']= df_final_table.groupby(['productCode', 'storeCode'])['itemCount'].shift()
df_final_table['saleLast2Month']= df_final_table.groupby(['productCode', 'storeCode'])['itemCount'].shift(2)
df_final_table['saleLast3Month']= df_final_table.groupby(['productCode', 'storeCode'])['itemCount'].shift(3)
df_final_table['saleLast4Month']= df_final_table.groupby(['productCode', 'storeCode'])['itemCount'].shift(4)
df_final_table['saleLast6Month']= df_final_table.groupby(['productCode', 'storeCode'])['itemCount'].shift(6)
df_final_table['saleLast12Month']= df_final_table.groupby(['productCode', 'storeCode'])['itemCount'].shift(12)
"""

for i in range (1, 13):
    df_final_table['saleLast%sMonth' %i]= df_final_table.groupby(['productCode', 'storeCode'])['itemCount'].shift(i)
    df_final_table['saleLast%sMonth' %i].fillna(0, inplace = True)

for i in range (1,13):
    
    df_final_table['max%sMonth' %i] = df_final_table.iloc[:,11:11+i ].max(axis=1)
    df_final_table['min%sMonth' %i] = df_final_table.iloc[:,11:11+i ].min(axis=1)
    df_final_table['mean%sMonth' %i] = df_final_table.iloc[:,11:11+i ].mean(axis=1)
    df_final_table['mad%sMonth' %i] = df_final_table.iloc[:,11:11+i ].mad(axis=1)
    df_final_table['std%sMonth' %i] = df_final_table.iloc[:,11:11+i ].std(axis=1)

"""
df_final_table["max4Month"] = df_final_table[['saleLast1Month', 'saleLast2Month', 'saleLast3Month', 'saleLast4Month']].max(axis=1)
df_final_table["min4Month"] = df_final_table[['saleLast1Month', 'saleLast2Month', 'saleLast3Month', 'saleLast4Month']].min(axis=1)
df_final_table["mean4Month"] = df_final_table[['saleLast1Month', 'saleLast2Month', 'saleLast3Month', 'saleLast4Month']].mean(axis=1)
df_final_table["mad4Month"] = df_final_table[['saleLast1Month', 'saleLast2Month', 'saleLast3Month', 'saleLast4Month']].mad(axis=1)
"""

start_time4 = time.time()


#Make a new item count column
#df_final_table["itemCount_fixed"] = df_final_table['itemCount'].clip(lower = 1)

df_final_table["growth_base"] = df_final_table["itemCount"].pct_change()

df_final_table["growth_base"].fillna(1, inplace = True)
df_final_table["growth1"] = df_final_table["growth_base"].shift()
df_final_table["growth2"] = df_final_table["growth_base"].shift(2)
df_final_table["growth3"] = df_final_table["growth_base"].shift(3)
df_final_table["growth4"] = df_final_table["growth_base"].shift(4)
df_final_table["growth5"] = df_final_table["growth_base"].shift(5)
df_final_table["growth6"] = df_final_table["growth_base"].shift(6)
df_final_table["growth12"] = df_final_table["growth_base"].shift(12)

print ("added growth column ", time.time() - start_time4)


#-------------------------------------------------------- CUT OFF 12 MONTH ----------------------------------------

no_of_month_sampled = 25
max_date = df_final_table["date"].max()
min_date = max_date - datetime.timedelta(days = no_of_month_sampled * 30 )
df_final_table = df_final_table[df_final_table["date"] >= min_date ] 
df_final_table.reset_index(drop = True, inplace = True)


df_final_table["valentine"] = (2 - df_final_table["month"]) %12
df_final_table["new_year"] = (1- df_final_table["month"]) %12
df_final_table["mid_autumn"] = (9 - df_final_table["month"]) %12


df_final_table["valentine"] = np.maximum(df_final_table["valentine"], 3)
df_final_table["new_year"] = np.maximum(df_final_table["new_year"], 3)
df_final_table["mid_autumn"] = np.maximum(df_final_table["mid_autumn"], 3)


df_final_table = df_final_table[['date',"month", "quarter", "year", 'storeCode', 'productCode', 'productType', 'price', "valentine", "new_year", "mid_autumn",
       'saleLast1Month', 'saleLast2Month', 'saleLast3Month', 'saleLast4Month', 'saleLast5Month', 'saleLast6Month', 'saleLast7Month', 'saleLast8Month', 'saleLast9Month', 'saleLast10Month', 'saleLast11Month', 'saleLast12Month',
       "growth1", "growth2", "growth3", "growth4", "growth5", "growth6", "growth12",
       "max1Month", "min1Month", "mean1Month", "mad1Month", "std1Month", "max2Month", "min2Month", "mean2Month", "mad2Month", "std2Month", "max4Month", "min4Month", "mean4Month", "mad4Month", "std4Month", "max5Month", "min5Month", "mean5Month", "mad5Month", "std5Month",
       "max3Month", "min3Month", "mean3Month", "mad3Month", "std3Month", "max6Month", "min6Month", "mean6Month", "mad6Month", "std6Month", "max12Month", "min12Month", "mean12Month", "mad12Month", "std12Month",
       'itemCount']]


#delete values before product generate sales
df_final_table.sort_values(by = ["productCode", "date"], ascending = [True, True], inplace = True)

#fill nan
df_final_table.set_index("date", inplace = True)
#df_final_table.interpolate(method = "time", inplace = True)
df_final_table.reset_index(inplace = True)

df_final_table.sort_values(by = ["productCode", "date", "itemCount"], ascending = [True, True, False], inplace = True)

current_product = "None"
current_price = 0
k= 0
dropped_index = []
while k <= df_final_table.shape[0]-1:
    #print (" k= ", k, "current product ", current_product, "current item count", df_final_table.iloc[k, -1])
    
    if df_final_table.iloc[k, -1] == 0:
        if  df_final_table.iloc[k, 5] != current_product:
            #print ("index = ", k," current prod ", str(current_product), " checking product ", str(df_final_table.iloc[k, 5]), " count ", str(df_final_table.iloc[k, -1])  )
            #df_final_table.drop(df_final_table.index[k],axis = 0, inplace = True)
            dropped_index.append(k)
            #print ("df size after drop ", df_final_table.shape[0])
            #continue
    else: 
        current_product = df_final_table.iloc[k, 5]
    k = k+1

print ("total dropped rows ", len(dropped_index))
df_final_table.drop(dropped_index,axis = 0, inplace = True)


df_final_table['price'].fillna(method = "ffill", inplace = True)

#OG anchor date is last sunday of 2014
min_date = datetime.datetime(2014, 12, 31, 0 ,0 , 0)
#mindate_ordinal = min_date.toordinal()

df_final_table['date'] = df_final_table["date"].apply(lambda x: int((x.year - 2014)*12 + x.month - 12 ))
df_final_table.sort_values(by = ["date", "productCode", "storeCode"], ascending = [True, True, True], inplace = True)

df_final_table.reset_index(drop = True, inplace = True)

#df_final_table.to_excel("processed_sale_data.xlsx")
print("--- %s seconds ---" % (time.time() - start_time))

#df_final_table.to_excel("all_sale_3107_month_new.xlsx")


# Create storage object with filename `processed_data`
data_store = pd.HDFStore('aha_20190919.h5')

# Put DataFrame into the object setting the key as 'preprocessed_df'
data_store['df_final_table'] = df_final_table
data_store.close()

"""
# Access data store
data_store = pd.HDFStore('processed_data.h5')

# Retrieve data using key
preprocessed_df = data_store['preprocessed_df']
data_store.close()
"""
del dataset, dataset_month, dropped_index, og_dataset
gc.collect()
