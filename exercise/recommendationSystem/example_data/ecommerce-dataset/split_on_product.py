import gc

import numpy as np # linear algebra
# import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

nRowsRead = None   # specify 'None' if want to read whole file
# events.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df2 = pd.read_csv('ecommerce-dataset/events.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'events.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')
# exercise/recommendationSystem/example_data/ecommerce-dataset/events.csv

prod_code = df2['itemid'].unique().tolist()

#for product that have integer itemCount
prod_to_del = []
tmp = df2.timestamp.max() - 3 * 2.592e+9
for i in range(len(prod_code)):
    if df2.loc[df2.itemid == prod_code[i]].timestamp.max() <= tmp:
        prod_to_del.append(prod_code[i])

for i in range(len(prod_to_del)):
    indexName = df2[df2.itemid == prod_to_del[i]].index
    # df2.drop(indexName, inplace = True)
