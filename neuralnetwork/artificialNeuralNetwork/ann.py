#artificial neural network

#data preprocessing
#importing library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv(r"~/machineLearning/Machine-Learning-A-Z-New/Machine Learning A-Z New/Part 8 - Deep Learning" +
                      r"/Section 39 - Artificial Neural Networks (ANN)/Churn_Modelling.csv")
#create independent matrix of X (all rows, all column except the last which is the result or y)
X = dataset.iloc[:, 3: 13].values
#create dependent variable vector y (0-based), 4 columns
y = dataset.iloc[:, 13].values

#encode categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
#labelencoder_X_1 =LabelEncoder()
#X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 =LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])
onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()

#overfitting the dataset
#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)    #random_state for consistency

#need to scale if features have different range
#feature scalling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)
#dummy variable scalling depend on context

#ANN
#import keras
import keras
from keras.models import Sequential
from keras.layers import Dense

#initializing the ANN
classifier = Sequential()
#put the input layer and first layer
classifier.add(Dense(units = 6, input_dim = 12, kernel_initializer = 'uniform', activation = 'relu'))
#2nd layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu'))
#output layer
classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))
#compiling ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

#fiting the training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)
#predicting the test set result
y_pred = classifier.predict(X_test)
y_pred_f = (y_pred > 0.5)

#making Confussion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_f)
#example use test_size = 0.2
