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
og_dataset = pd.read_excel('beemart_20190802.xlsx', header = None)
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
    stores.append(dataset.loc[dataset.storeCode == store_code[i]].reset_index())
    stores[i].drop('index', 1, inplace = True)

store_inactive = [stores[4].storeCode[0], stores[8].storeCode[0], stores[10].storeCode[0]]

for i in [4, 8, 10]:
    a = stores[i].groupby(['date']).itemCount.sum()
    for j in range(len(a._index) - 1):
        if a._index[j] + datetime.timedelta(days = 30) < a._index[j + 1]:
            print("Shop %(sh)s inactive for %(t)s from %(n)s" %{'sh':stores[i].storeCode[0], 't':(a._index[j + 1] - a._index[j]), 'n':(a._index[j])})
    print("End of shop %s" %stores[i].storeCode[0])
    print()

del og_dataset, a, start_time, end_time, store_code
gc.collect()
