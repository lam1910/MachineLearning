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


#PREDICT_DATE = datetime.datetime(year = 2019, Week = 6, day = 30)
start_time = time.time() 
#og_dataset = pd.read_excel('thongke_whatev_20190823.xlsx', header = None)
#using old data
og_dataset = pd.read_excel('beemart_20190802.xlsx', header = None)
#newer data: beemart_20190903.xlsx (do not have locally)

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
dataset_week = dataset.groupby([pd.Grouper(key='date', freq='W'), pd.Grouper(key='productCode'), pd.Grouper(key='productType'), pd.Grouper(key='storeCode')]).agg({"price":"mean","itemCount":"sum", "revenue": "sum"})
dataset_week.reset_index(drop = False, inplace = True)

columns_to_encode = ['productCode', 'productType', 'storeCode']

dataset_week[columns_to_encode] = dataset_week[columns_to_encode].astype(str)

#dataset_week[["productCode","productType","storeCode"]] = dataset_week[["productCode","productType","storeCode"]].astype(str)


no_of_week_sampled = 78
max_date = dataset_week["date"].max()
min_date = max_date - datetime.timedelta(days = no_of_week_sampled * 7)
dataset_week = dataset_week[dataset_week["date"] >= min_date ] 
dataset_week = dataset_week[dataset_week["date"] <= max_date ]
dataset_week.reset_index(drop = True, inplace = True)


#delete inactive stores
storelist = ["14017", "14018",  "14165", "36040",  "14015", "16526"]
#, "14016", "50875", "88087"
dataset_week= dataset_week[dataset_week.storeCode.isin(storelist)]
dataset_week.reset_index(drop = True, inplace = True)


"""
k= 0
dropped_index = []
while k <= dataset_week.shape[0]-1:
    #print ("index = ", k," store ", dataset_week.iloc[k, 3], " year ", dataset_week.iloc[k,0].year)
    if (dataset_week.iloc[k, 3] == '14016')  and  (dataset_week.iloc[k,0].year < 2019):
        print ("index = ", k," store ", dataset_week.iloc[k, 3], " year ", dataset_week.iloc[k,0].year)
        #df_final_table.drop(df_final_table.index[k],axis = 0, inplace = True)
        dropped_index.append(k)
        #print ("df size after drop ", df_final_table.shape[0])
        #continue
    k = k+1

print ("total dropped rows ", len(dropped_index))
dataset_week.drop(dropped_index,axis = 0, inplace = True)
"""


#dataset_week= dataset_week[dataset_week["productCode"]== 'VNL00325']
"""
dataset_week= dataset_week[dataset_week["storeCode"]== '36040' ]
"""

sale_by_store = dataset_week.groupby(["storeCode"]).agg({"itemCount":"sum", "revenue": "sum"})
sale_by_product = dataset_week.groupby(["productCode"]).agg({"itemCount":"sum", "revenue": "sum"})
sale_by_type = dataset_week.groupby(["productType"]).agg({"itemCount":"sum", "revenue": "sum"})

max_date_2018 = datetime.datetime(year = 2019, month = 8, day = 1)
min_date_2018 = datetime.datetime(year = 2015, month = 12, day = 31)
total_sale_by_week = dataset_week#[dataset_week["productCode"]== 'VNL00325']
#total_sale_by_Week = total_sale_by_Week[total_sale_by_Week["storeCode"]== '14017' ]
total_sale_by_week = total_sale_by_week.groupby(["date"]).agg({"itemCount":"sum", "revenue": "sum"})
total_sale_by_week.reset_index(inplace = True)
total_sale_by_week = total_sale_by_week[(total_sale_by_week["date"]> min_date_2018) ]
total_sale_by_week = total_sale_by_week[(total_sale_by_week["date"]< max_date_2018) ]


#REMOVE FUCKING BUTTER 
"""
df_product_sale = dataset_week.groupby(["productCode"]).agg({"itemCount":"sum"})
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

for row_index, row in dataset_week.iterrows():
    if dataset_week.iloc[row_index, 1] in bad_product:
        delete_count +=1
        dataset_week.iloc[row_index, 4] = dataset_week.iloc[row_index, 4] * 1000
        dataset_week.iloc[row_index, 5] = float(dataset_week.iloc[row_index, 5]) / 1000
        #print ("FUCKING BUTTER ", dataset_week.iloc[row_index,1], "total remove ", delete_count )
        print ("FUCKING BUTTER ", dataset_week.iloc[row_index,1], "total remove ", delete_count )
        print ("new price ", dataset_week.iloc[row_index, 4], " new sale ", float(dataset_week.iloc[row_index, 5]))
        deleted_index.append(row_index)
#dataset_week.drop(deleted_index,axis = 0, inplace = True)
#dataset_week.reset_index(drop = True, inplace = True)

"""

time1= time.time()
product_count = dataset_week.groupby(["productCode"]).agg({"productType":"first"})
product_count.reset_index(inplace = True)
#base_list = base_list.to_frame()
#base_list["total_count"] = base_list["productCode"].apply(lambda x: product_count[product_count["productCode"] == x]["itemCount"].sum())
store_count = dataset_week.groupby(["storeCode"]).agg({"itemCount":"sum"})
store_count.reset_index(inplace = True)
date_count = dataset_week.groupby(["date"]).agg({"itemCount":"count"})
date_count.reset_index(inplace = True)

# Make sure all store and products are present
df_final_table = pd.MultiIndex.from_product([date_count["date"], product_count["productCode"], store_count["storeCode"]], names=['date','productCode', 'storeCode']).to_frame()
dataset_week.set_index(["date", "productCode", "storeCode"], inplace = True)

df_final_table = df_final_table.join(dataset_week, how = 'left')
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

del(dataset, dataset_week, og_dataset, columns_to_encode, no_of_week_sampled, max_date, min_date, storelist, sale_by_product, sale_by_store, sale_by_type, max_date_2018, min_date_2018, total_sale_by_week, date_count, product_count, store_count)
gc.collect()


#add more row for week with no sale
"""
df_price = dataset_week.set_index(
    ["storeCode", "productCode", "productType", "date"])[["price"]].unstack(
        level=-1)
df_price.columns = df_price.columns.get_level_values(1)
df_price.reset_index()
df_final_price = pd.pivot_table(df_price,columns = ["storeCode","productCode","productType"]).reset_index()

df_final_table = df_final_price
df_final_price.columns = ["date", "storeCode", "productCode", "productType", "price"]
del df_price, df_final_price
gc.collect()

df_count = dataset_week.set_index(
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


#---------------------------- CALCULATE PAST 1 2 3 AND 4 Week SALES -----------------------------------------------------
print("Adding past weeks sale as new columns: ")


df_final_table["month"] = df_final_table["date"].dt.month
df_final_table["quarter"] = df_final_table["date"].dt.quarter
df_final_table["year"] = df_final_table["date"].dt.year




#df_final_table.iloc[-1, 5] = 9999999999
"""
df_final_table['saleLast1Week']= df_final_table.groupby(['productCode', 'storeCode'])['itemCount'].shift()
df_final_table['saleLast2Week']= df_final_table.groupby(['productCode', 'storeCode'])['itemCount'].shift(2)
df_final_table['saleLast3Week']= df_final_table.groupby(['productCode', 'storeCode'])['itemCount'].shift(3)
df_final_table['saleLast4Week']= df_final_table.groupby(['productCode', 'storeCode'])['itemCount'].shift(4)
df_final_table['saleLast6Week']= df_final_table.groupby(['productCode', 'storeCode'])['itemCount'].shift(6)
df_final_table['saleLast12Week']= df_final_table.groupby(['productCode', 'storeCode'])['itemCount'].shift(12)
"""

for i in range (1, 53):
    df_final_table['saleLast%sWeek' %i]= df_final_table.groupby(['productCode', 'storeCode'])['itemCount'].shift(i)
    df_final_table['saleLast%sWeek' %i].fillna(0, inplace = True)

for i in [2,3,4,6,8,12,24,52]:
    
    df_final_table['max%sWeek' %i] = df_final_table.iloc[:,11:11+i ].max(axis=1)
    df_final_table['min%sWeek' %i] = df_final_table.iloc[:,11:11+i ].min(axis=1)
    df_final_table['mean%sWeek' %i] = df_final_table.iloc[:,11:11+i ].mean(axis=1)
    df_final_table['mad%sWeek' %i] = df_final_table.iloc[:,11:11+i ].mad(axis=1)
    df_final_table['std%sWeek' %i] = df_final_table.iloc[:,11:11+i ].std(axis=1)

"""
df_final_table["max4Week"] = df_final_table[['saleLast1Week', 'saleLast2Week', 'saleLast3Week', 'saleLast4Week']].max(axis=1)
df_final_table["min4Week"] = df_final_table[['saleLast1Week', 'saleLast2Week', 'saleLast3Week', 'saleLast4Week']].min(axis=1)
df_final_table["mean4Week"] = df_final_table[['saleLast1Week', 'saleLast2Week', 'saleLast3Week', 'saleLast4Week']].mean(axis=1)
df_final_table["mad4Week"] = df_final_table[['saleLast1Week', 'saleLast2Week', 'saleLast3Week', 'saleLast4Week']].mad(axis=1)
"""

start_time4 = time.time()


#Make a new item count column
#df_final_table["itemCount_fixed"] = df_final_table['itemCount'].clip(lower = 1)

df_final_table["growth_base"] = df_final_table["itemCount"].pct_change()
df_final_table["growth_base"].fillna(1, inplace = True)

for i in [1,2,3,4,5,6,8,12,24,52]:    
    df_final_table['growth%s' %i] = df_final_table["growth_base"].shift(i)
    
print ("added growth column ", time.time() - start_time4)


#-------------------------------------------------------- CUT OFF 12 Week ----------------------------------------

no_of_week_sampled = 100
max_date = df_final_table["date"].max()
min_date = max_date - datetime.timedelta(days = no_of_week_sampled * 7 )
df_final_table = df_final_table[df_final_table["date"] >= min_date ] 
df_final_table.reset_index(drop = True, inplace = True)


df_final_table["valentine"] = (2 - df_final_table["month"]) %12
df_final_table["new_year"] = (1- df_final_table["month"]) %12
df_final_table["mid_autumn"] = (9 - df_final_table["month"]) %12


df_final_table["valentine"] = np.maximum(df_final_table["valentine"], 3)
df_final_table["new_year"] = np.maximum(df_final_table["new_year"], 3)
df_final_table["mid_autumn"] = np.maximum(df_final_table["mid_autumn"], 3)


df_final_table = df_final_table[['date',"month", "quarter", "year", 'storeCode', 'productCode', 'productType', 'price', "valentine", "new_year", "mid_autumn",
       'saleLast1Week', 'saleLast2Week', 'saleLast3Week', 'saleLast4Week', 'saleLast5Week', 'saleLast6Week', 'saleLast7Week', 'saleLast8Week', 'saleLast9Week', 'saleLast10Week', 'saleLast11Week', 'saleLast12Week',
       'saleLast24Week', 'saleLast52Week',
       "growth1", "growth2", "growth3", "growth4", "growth5", "growth8", "growth12", "growth24", "growth52",
       "max2Week", "min2Week", "mean2Week", "mad2Week", "std2Week", "max4Week", "min4Week", "mean4Week", "mad4Week", "std4Week", "max8Week", "min8Week", "mean8Week", "mad8Week", "std8Week",
       "max3Week", "min3Week", "mean3Week", "mad3Week", "std3Week", "max6Week", "min6Week", "mean6Week", "mad6Week", "std6Week", "max12Week", "min12Week", "mean12Week", "mad12Week", "std12Week",
       "max24Week", "min24Week", "mean24Week", "mad24Week", "std24Week", "max52Week", "min52Week", "mean52Week", "mad52Week", "std52Week",
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
min_date = datetime.datetime(2014, 12, 28, 0 ,0 , 0)
#mindate_ordinal = min_date.toordinal()

df_final_table['date'] = df_final_table["date"].apply(lambda x: int((x-min_date).days /7)  )
df_final_table.sort_values(by = ["date", "productCode", "storeCode"], ascending = [True, True, True], inplace = True)

df_final_table.reset_index(drop = True, inplace = True)

#df_final_table.to_excel("processed_sale_data.xlsx")
print("--- %s seconds ---" % (time.time() - start_time))

#df_final_table.to_excel("all_sale_3107_Week_new.xlsx")


# Create storage object with filename `processed_data`
data_store = pd.HDFStore('beemart_0208_week.h5')

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
del dataset, dataset_week, dropped_index, og_dataset
gc.collect()
