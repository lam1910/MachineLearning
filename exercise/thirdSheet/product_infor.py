# Importing the libraries
import numpy as np
import pandas as pd
from pandas import ExcelWriter
import matplotlib.pyplot as plt
import time
import gc


print("________________________________")
print("Start getting data about time each product was sold.")
start_time = time.time()
og_dataset = pd.read_excel('MASS_corele_dulieu_20190919.xlsx', header = None)
og_dataset.columns = ["date", "productCode", "productType", "storeCode", "price", "itemCount"]

dataset = og_dataset.iloc[:, :]
dataset.reset_index(inplace = True)
dataset.drop(columns = "index", axis = 1, inplace = True)
dataset.iloc[:, 0] = pd.to_datetime(dataset.iloc[:, 0], format = "%Y%m%d" )


#put different dataframe for different productCode

prod_code = dataset['productCode'].unique().tolist()

#for product that have integer itemCount
prods = []
for i in range(len(prod_code)):
    prods.append(dataset.loc[dataset.productCode == prod_code[i]].reset_index())

product_infor = list()
for prod in prods:
    product_infor.append([prod.productCode[0], prod.date.min(), prod.date.max()])

product_infor = np.array(product_infor)

for i in range(product_infor.shape[0]):
    product_infor[:, 1][i] = product_infor[:, 1][i].strftime("%Y%m%d")
    product_infor[:, 2][i] = product_infor[:, 2][i].strftime("%Y%m%d")

products = pd.DataFrame(data = {'productCode':product_infor[:, 0] , 'startDate':product_infor[:, 1]
                                , 'endDate':product_infor[:, 2]}, dtype = str)



end_time = time.time()
print("Done! Total time execute: %s seconds" %(end_time - start_time))


print()
print("________________________________")
print("Start writing data about time each product was sold.")
start_write = time.time()
with ExcelWriter(path = "product_infor.xlsx") as writer:
    products.to_excel(writer, sheet_name = 'Sheet0', index = False)
end_write = time.time()

print("Done! Time execute: %s seconds" %(end_write - start_write))


del start_time, end_time, start_write, end_write, og_dataset, dataset, prod_code, prods, product_infor
gc.collect()


