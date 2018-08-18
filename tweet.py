# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 02:10:52 2018

@author: Heller
"""
import time
import numpy as np
from skll.metrics import kappa
from nltk.corpus import stopwords
from pandas.io.parsers import read_csv
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from naivebayes import NaiveBayesTextClassifier
from sklearn import svm
from sklearn import preprocessing
from sklearn.svm import LinearSVC
def non_shuffling_train_test_split(X, y, test_size=0.2):
    i = int((1 - test_size) * X.shape[0]) + 1
    X_train, X_test = np.split(X, [i])
    y_train, y_test = np.split(y, [i])
    return X_train, X_test, y_train, y_test
print("> Read train data")
train_data = read_csv('train.csv')
print("> Init classifier")
start_time = time.time()
classifier = NaiveBayesTextClassifier(
    categories=['1','0','t'],
    min_df=1,
    lowercase=True,
    # 127 English stop words
    stop_words=stopwords.words('english')
)

print("> Split data to test and train")
train_docs, test_docs, train_classes, test_classes = non_shuffling_train_test_split(
        train_data['tweet'], train_data['label'])
train_docs = train_docs.fillna('1')
train_classes = train_classes.fillna('1')
print (train_docs.isnull().any())
print (train_classes.isnull().any())
print (type(train_docs))
print("> Train classifier")
classifier.train(train_docs, train_classes)
total_docs = len(train_docs)
print("-" * 42)
print("Total", total_docs," tweets")
print(
    "Number of words", classifier.bag.shape[1]," words"
)
print(
    "Parse time", time.time() - start_time, "seconds"
)
print("-" * 42)

# -------------- Classify --------------- #

print("> Start classify data")
start_time = time.time()
test_docs = test_docs.fillna('1')
test_classes = test_classes.fillna('1')
predicted_classes = classifier.classify(test_docs)
print ((predicted_classes),(test_classes))
print(classification_report(test_classes, predicted_classes))
print('-' * 42)
print("Test data size", len(test_classes),"articles")
print(
        "Accuracy", 100 * accuracy_score(test_classes, predicted_classes),"%")
end_time = time.time()
print(
    "Computation time", end_time - start_time, "seconds"
)
print('-' * 42)