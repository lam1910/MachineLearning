import numpy as np
import pandas as pd
from statistics import stdev
from statistics import StatisticsError


dataset = pd.read_excel(io = r"~/VSCode/Python/machinelearning/exercise/processed_sale_data.xlsx", sheet_name = 0
                        , usecols = [1, 2, 3, 4, 5, 6]
                        , dtype = {'storeCode': str, 'productCode': str, 'productType': str})

prod_code = dataset['productCode'].unique().tolist()
prods = []
store_in_prods = []
for i in range(len(prod_code)):
    prods.append(dataset.loc[dataset.productCode == prod_code[i]])
    store_code = prods[i]['storeCode'].unique()
    for k in range(len(store_code)):
        tmp = prods[i].loc[dataset.storeCode == store_code[k]].values
        tmp = np.append(arr = tmp, values = np.ones((np.size(tmp, 0), 6)).astype(int), axis = 1)
        store_in_prods.append(tmp)


for tmp in store_in_prods:
    tmp[:, 6] = tmp[:, 6] * max(tmp[:, 5])
    tmp[:, 7] = tmp[:, 7] * min(tmp[:, 5])
    try:
        tmp[:, 11] = tmp[:, 11] * stdev(tmp[:, 5])
    except StatisticsError:
        tmp[:,11] = tmp[:, 11] * 0


for tmp in store_in_prods:
    tmp[0, [8, 9, 10]] = tmp[0, [8, 9, 10]] * 0
    try:
        tmp[1, [9, 10]] = tmp[1, [9, 10]] * 0
        tmp[2, [10]] = tmp[2, [10]] * 0
        tmp[3, [10]] = tmp[3, [10]] * 0
    except IndexError:
        continue

for tmp in store_in_prods:
    if np.size(tmp, 0) == 1:
        continue
    else:
        for i in range(1, (np.size(tmp, 0))):
            tmp[i, 8] = tmp[(i - 1), 5]
            if tmp[i, 9] == 0:
                continue
            elif tmp[i, 10] == 0:
                tmp[i, 9] = tmp[(i - 2), 5]
                continue
            else:
                tmp[i, 9] = tmp[(i - 2), 5]
                tmp[i, 10] = tmp[(i - 4), 5]


to_new_dataset = np.concatenate(store_in_prods)

to_new_excel = pd.DataFrame({'date':to_new_dataset[:, 0], 'storeCode':to_new_dataset[:, 1]
                            , 'productCode':to_new_dataset[:, 2], 'productType':to_new_dataset[:, 3]
                            , 'price':to_new_dataset[:, 4], 'itemCount':to_new_dataset[:, 5]
                            , 'maxItemCount':to_new_dataset[:, 6], 'minItemCount':to_new_dataset[:, 7]
                            , 'itemCountLast1Week':to_new_dataset[:, 8], 'itemCountLast2Weeks':to_new_dataset[:, 9]
                            , 'itemCountLast4Weeks':to_new_dataset[:, 10], 'stdev':to_new_dataset[:, 11]})

to_new_excel.to_csv(path_or_buf = r"~/VSCode/Python/machinelearning/exercise/processed_sale_data_ver2.csv")