#nautural language processing

#importing library

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset = pd.read_csv(filepath_or_buffer = r"~/machineLearning/Machine-Learning-A-Z-New/Machine Learning A-Z New" +
                        r"/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing" +
                        r"/Restaurant_Reviews.tsv"
                      , delimiter = '\t', quoting = 3)

#cleaning the text
import re
import nltk
#if not download then uncomment the next line. If already download or uncomment led to error comment the next line
#nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(1000):
    #retain only text
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    #change to lower case
    review = review.lower()
    #remove the non-significant word
    review = review.split()
    #review = [word for word in review if not word in set(stopwords.words('english'))]
    #stemming
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #join back to create a string of basic review
    review = ' '.join(review)
    corpus.append(review)

#creating the bag of word
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values


#overfitting the dataset
#splitting the dataset into training set and test set
from sklearn.model_selection import train_test_split
#random_state for consistency
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


#naive bayes
#fiting naive bayes model to dataset
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

#predicting the test set result
y_pred = classifier.predict(X_test)

#making Confussion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


#K-NN
#fiting K-NN model to dataset
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

#predicting the test set result
y_pred = classifier.predict(X_test)

#making Confussion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
