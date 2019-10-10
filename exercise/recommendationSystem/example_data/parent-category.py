import pandas as pd
import numpy as np


nRowsRead = None    # specify 'None' if want to read whole file
# category_tree.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('ecommerce-dataset/category_tree.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'category_tree.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
# exercise/recommendationSystem/example_data/ecommerce-dataset/category_tree.csv

X = df1.iloc[:, :].values

rollback = pd.DataFrame({'categoryid': X[:, 0], 'parentid': X[:, 1]})

flatten = list()

for row in X:
    try:
        tmp = str(int(row[0])) + ' ' + str(int(row[1]))
    except ValueError:
        tmp = str(int(row[0])) + ' ' + str(row[1])
    
    flatten.append(tmp)

df1['flattened'] = flatten

rollback.to_csv(path_or_buf = 'ecommerce-datset/category-tree.csv', sep = ',')
# exercise/recommendationSystem/example_data/ecommerce-dataset/category_tree.csv

df1.to_csv(path_or_buf = 'ecommerce-datset/category-tree-ver2.csv', sep = ',')
# exercise/recommendationSystem/example_data/ecommerce-dataset/category_tree-ver2.csv
