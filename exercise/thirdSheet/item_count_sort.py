# Importing the libraries
import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import time
import gc

from pandas import ExcelWriter

print("________________________________")
print("Start getting data about item count.")
start_time = time.time()
og_dataset = pd.read_excel('MASS_corele_dulieu_20190919.xlsx', header = None)
og_dataset.columns = ["date", "productCode", "productType", "storeCode", "price", "itemCount"]

dataset = og_dataset.iloc[:, :]
dataset.reset_index(inplace = True)
dataset.drop(columns = "index", axis = 1, inplace = True)
dataset.iloc[:, 0] = pd.to_datetime(dataset.iloc[:, 0], format = "%Y%m%d" )


"""dataset = dataset.iloc[:, [0, 1, 3, 4, 5]]
prod_code = dataset['productCode'].unique().tolist()
store_code = dataset['storeCode'].unique().tolist()

prods = []
for i in range(len(prod_code)):
    prods.append(dataset.loc[dataset.productCode == prod_code[i]].reset_index(drop = True))

prod_stores = []
for prod in prods:
    for i in range(len(store_code)):
        if store_code[i] in prod.storeCode.array:
            prod_stores.append(prod.loc[prod.storeCode == store_code[i]].reset_index(drop = True))
        else:
            continue
"""

new_dataset = dataset.sort_values(by = "itemCount", ascending = False)

large_sale = new_dataset.loc[new_dataset.itemCount >= 100]
small_sale = new_dataset.loc[new_dataset.itemCount < 100]

end_time = time.time()
print("Finish getting data about item count. Time %s second(s)." %(end_time - start_time))
print("________________________________")

print()
print("________________________________")
print("Start writing data about time each product was sold.")
start_write = time.time()
with ExcelWriter(path = "second_solution_sort_by_itemCount.xlsx") as writer:
    new_dataset.to_excel(writer, sheet_name = 'Sorted by itemCOunt', index = False)
    large_sale.to_excel(writer, sheet_name = 'itemCount>=100', index = False)
    small_sale.to_excel(writer, sheet_name = 'itemCount<100', index = False)
end_write = time.time()

print("Done! Time execute: %s seconds" %(end_write - start_write))


"""tmp = prod_stores[42]
plt.bar(tmp.date.unique(), tmp.groupby(['date']).sum().itemCount, color = 'red')
plt.title("%(strC)s %(prdC)s" %{'strC': tmp.storeCode[0], 'prdC': tmp.productCode[0]})"""