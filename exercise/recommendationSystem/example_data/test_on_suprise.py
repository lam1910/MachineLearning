import pandas as pd     # data processing, CSV file I/O (e.g. pd.read_csv)
import gc

from surprise import SVD
from surprise.model_selection import train_test_split
from surprise.dataset import Dataset
from surprise import Reader
from surprise import accuracy


nRowsRead = None # specify 'None' if want to read whole file
# events.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df2 = pd.read_csv('ecommerce-dataset/events.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'events.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')
#exercise/recommendationSystem/example_data/ecommerce-dataset/events.csv

dataset = df2.replace(['view', 'addtocart', 'transaction'], [0, 1, 2])
# addtocart: 1
# transaction: 2
# view: 0

dataset = dataset.iloc[:, [1, 3, 2]]

del df2
gc.collect()


reader = Reader(rating_scale=(0, 2))

true_dataset = Dataset.load_from_df(dataset, reader)

algo = SVD()
trainset, testset = train_test_split(true_dataset, test_size=.25)
algo.fit(trainset)
predictions = algo.test(testset)

for prediction in predictions:
    print(prediction)

print("RMSE: %(r)s and MSE: %(m)s" %{'r': accuracy.rmse(predictions), 'm': accuracy.mse(predictions)})
