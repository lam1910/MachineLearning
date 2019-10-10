import pandas as pd     # data processing, CSV file I/O (e.g. pd.read_csv)
import gc


# from surprise import SVD
# from surprise.model_selection import train_test_split
# from surprise.dataset import Dataset
# from surprise import Reader
# from surprise import accuracy


nRowsRead = None # specify 'None' if want to read whole file
# events.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df2 = pd.read_csv('ecommerce-dataset/events.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'events.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')
#exercise/recommendationSystem/example_data/ecommerce-dataset/events.csv


df2.timestamp = pd.to_datetime(arg = df2.timestamp, unit = 'ms', origin = 'unix')
df2.sort_values(by = 'timestamp', axis = 0, inplace = True)
df2.reset_index(drop = True, inplace = True)

vit_code = df2['visitorid'].unique().tolist()

vits = []
for i in range(len(vit_code)):
    vits.append(df2.loc[df2.itemid == vit_code[i]].reset_index())
