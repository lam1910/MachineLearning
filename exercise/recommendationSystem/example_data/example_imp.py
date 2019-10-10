# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt # plotting
import gc

import numpy as np # linear algebra
# import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

"""
for dirname, _, filenames in os.walk('exercise/recommendationSystem/example_data/ecommerce-dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))"""

"""

# ../../../../
def plotPerColumnDistribution(df, nGraphShown, nGraphPerRow):
    nunique = df.nunique()
    df = df[[col for col in df if nunique[col] > 1 and nunique[col] < 50]] # For displaying purposes, pick columns that have between 1 and 50 unique values
    nRow, nCol = df.shape
    columnNames = list(df)
    nGraphRow = (nCol + nGraphPerRow - 1) / nGraphPerRow
    plt.figure(num = None, figsize = (6 * nGraphPerRow, 8 * nGraphRow), dpi = 80, facecolor = 'w', edgecolor = 'k')
    for i in range(min(nCol, nGraphShown)):
        plt.subplot(nGraphRow, nGraphPerRow, i + 1)
        columnDf = df.iloc[:, i]
        if (not np.issubdtype(type(columnDf.iloc[0]), np.number)):
            valueCounts = columnDf.value_counts()
            valueCounts.plot.bar()
        else:
            columnDf.hist()
        plt.ylabel('counts')
        plt.xticks(rotation = 90)
        plt.title(f'{columnNames[i]} (column {i})')
    plt.tight_layout(pad = 1.0, w_pad = 1.0, h_pad = 1.0)
    plt.show()


def plotCorrelationMatrix(df, graphWidth):
    filename = df.dataframeName
    df = df.dropna('columns') # drop columns with NaN
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    if df.shape[1] < 2:
        print(f'No correlation plots shown: The number of non-NaN or constant columns ({df.shape[1]}) is less than 2')
        return
    corr = df.corr()
    plt.figure(num=None, figsize=(graphWidth, graphWidth), dpi=80, facecolor='w', edgecolor='k')
    corrMat = plt.matshow(corr, fignum = 1)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.gca().xaxis.tick_bottom()
    plt.colorbar(corrMat)
    plt.title(f'Correlation Matrix for {filename}', fontsize=15)
    plt.show()


# Scatter and density plots
def plotScatterMatrix(df, plotSize, textSize):
    df = df.select_dtypes(include =[np.number]) # keep only numerical columns
    # Remove rows and columns that would lead to df being singular
    df = df.dropna('columns')
    df = df[[col for col in df if df[col].nunique() > 1]] # keep columns where there are more than 1 unique values
    columnNames = list(df)
    if len(columnNames) > 10: # reduce the number of columns for matrix inversion of kernel density plots
        columnNames = columnNames[:10]
    df = df[columnNames]
    ax = pd.plotting.scatter_matrix(df, alpha=0.75, figsize=[plotSize, plotSize], diagonal='kde')
    corrs = df.corr().values
    for i, j in zip(*plt.np.triu_indices_from(ax, k = 1)):
        ax[i, j].annotate('Corr. coef = %.3f' % corrs[i, j], (0.8, 0.2), xycoords='axes fraction', ha='center', va='center', size=textSize)
    plt.suptitle('Scatter and Density Plot')
    plt.show()"""


"""
nRowsRead = 1000    # specify 'None' if want to read whole file
# category_tree.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df1 = pd.read_csv('ecommerce-dataset/category_tree.csv', delimiter=',', nrows = nRowsRead)
df1.dataframeName = 'category_tree.csv'
nRow, nCol = df1.shape
print(f'There are {nRow} rows and {nCol} columns')
# exercise/recommendationSystem/example_data/ecommerce-dataset/category_tree.csv

plotPerColumnDistribution(df1, 10, 5)
plotCorrelationMatrix(df1, 8)
plotScatterMatrix(df1, 6, 15)"""


nRowsRead = None   # specify 'None' if want to read whole file
# events.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df2 = pd.read_csv('ecommerce-dataset/events.csv', delimiter=',', nrows = nRowsRead)
df2.dataframeName = 'events.csv'
nRow, nCol = df2.shape
print(f'There are {nRow} rows and {nCol} columns')
# exercise/recommendationSystem/example_data/ecommerce-dataset/events.csv

"""
plotPerColumnDistribution(df2, 10, 5)
plotCorrelationMatrix(df2, 8)
plotScatterMatrix(df2, 12, 10)"""
"""
nRowsRead = None    # specify 'None' if want to read whole file
# item_properties_part1.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df3 = pd.read_csv('ecommerce-dataset/item_properties_part1.csv', delimiter=',', nrows = nRowsRead)
df3.dataframeName = 'item_properties_part1.csv'
nRow, nCol = df3.shape
print(f'There are {nRow} rows and {nCol} columns')
# exercise/recommendationSystem/example_data/ecommerce-dataset/item_properties_part1.csv

plotPerColumnDistribution(df3, 10, 5)
plotCorrelationMatrix(df3, 8)
plotScatterMatrix(df3, 6, 15)

nRowsRead = None    # specify 'None' if want to read whole file
# item_properties_part1.csv may have more rows in reality, but we are only loading/previewing the first 1000 rows
df4 = pd.read_csv('ecommerce-dataset/item_properties_part2.csv', delimiter=',', nrows = nRowsRead)
df4.dataframeName = 'item_properties_part1.csv'
nRow, nCol = df4.shape
print(f'There are {nRow} rows and {nCol} columns')
# exercise/recommendationSystem/example_data/ecommerce-dataset/item_properties_part2.csv

plotPerColumnDistribution(df4, 10, 5)
plotCorrelationMatrix(df4, 8)
plotScatterMatrix(df4, 6, 15)

full_dataset = df3.append(df4, ignore_index = True, sort = True)"""


dataset = df2.replace(['view', 'addtocart', 'transaction'], [0, 1, 2])
# addtocart: 1
# transaction: 2
# view: 0

dataset = dataset.iloc[:, [1, 3, 2]]

del df2
gc.collect()

from surprise import SVD
from surprise.model_selection import train_test_split
from surprise.dataset import Dataset
from surprise import Reader
from surprise import accuracy

reader = Reader(rating_scale=(0, 2))

true_dataset = Dataset.load_from_df(dataset, reader)

algo = SVD()
trainset, testset = train_test_split(true_dataset, test_size=.25)
algo.fit(trainset)
predictions = algo.test(testset)

for prediction in predictions:
    print(prediction)

print("RMSE: %(r)s, MSE: %(m)s and FCP: %(f)s." %{'r': accuracy.rmse(predictions), 'm': accuracy.mse(predictions), 'f': accuracy.fcp(predictions)})
