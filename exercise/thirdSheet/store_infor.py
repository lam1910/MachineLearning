# Importing the libraries
import datetime

import numpy as np
import pandas as pd
from pandas import ExcelWriter
import matplotlib.pyplot as plt

import time
import gc


print("________________________________")
print("Start getting data about each store.")
start_time = time.time()
og_dataset = pd.read_excel('MASS_corele_dulieu_20190919.xlsx', header = None)
og_dataset.columns = ["date", "productCode", "productType", "storeCode", "price", "itemCount"]

dataset = og_dataset.iloc[:, :]
dataset.reset_index(inplace = True)
dataset.drop(columns = "index", axis = 1, inplace = True)
dataset.iloc[:, 0] = pd.to_datetime(dataset.iloc[:, 0], format = "%Y%m%d" )
end_time = time.time()
print("Done! Total time execute: %s seconds" %(end_time - start_time))

print()
print("________________________________")
print("Processing")

store_code = dataset['storeCode'].unique().tolist()

stores = []
for i in range(len(store_code)):
    stores.append(dataset.loc[dataset.storeCode == store_code[i]].reset_index(drop = True))

store_infor = list()
for store in stores:
    store_infor.append([store.storeCode[0], store.date.min(), store.date.max(), store.itemCount.sum()])
"""
store_infor = np.array(store_infor)

for i in range(store_infor.shape[0]):
    store_infor[:, 1][i] = store_infor[:, 1][i].strftime("%Y%m%d")
    store_infor[:, 2][i] = store_infor[:, 2][i].strftime("%Y%m%d")

products = pd.DataFrame(data = {'storeCode':store_infor[:, 0] , 'startDate':store_infor[:, 1]
                                , 'endDate':store_infor[:, 2], 'totalSale':store_infor[:, 3]}, dtype = str)

print()
print("________________________________")
print("Start writing data about time each product was sold.")
start_write = time.time()
with ExcelWriter(path = "MASS_corele_store_infor_20190919.xlsx") as writer:
    products.to_excel(writer, sheet_name = 'Sheet0', index = False)
end_write = time.time()

print("Done! Time execute: %s seconds" %(end_write - start_write))


del start_time, end_time, start_write, end_write, og_dataset, dataset, store_code, stores, store_infor
gc.collect()
"""
tmp = stores[50]
plt.bar(tmp.date.unique(), tmp.groupby(['date']).sum().itemCount, color = 'red')
plt.title("%s" %tmp.storeCode[0])

