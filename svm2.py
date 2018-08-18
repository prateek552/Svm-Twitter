
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
data=pd.read_csv('train.csv')
data['tweet']=data['tweet'].fillna('1')
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data['tweet'])
X_train_counts.shape
from sklearn.feature_extraction.text import TfidfTransformer
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape
from sklearn.pipeline import Pipeline
import numpy as np
from sklearn.metrics import classification_report,accuracy_score
test=pd.read_csv('test.csv')
from sklearn.linear_model import SGDClassifier
text_clf_svm = Pipeline([('vect', CountVectorizer()), ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, n_iter=5, random_state=42))])

data['label']=data['label'].fillna('1')
print (data['label'])
text_clf_svm = text_clf_svm.fit(data['tweet'], (data['label']))
test['tweet']=test['tweet'].fillna('1')
predicted_svm = text_clf_svm.predict(test['tweet'])
test['label']=test['label'].fillna('1')
print ("*"*45)
print (predicted_svm)
print ("*"*45)
print (classification_report(test['label'], predicted_svm))
np.mean(predicted_svm == test['label'])
print("Accuracy", 100 * accuracy_score(test['label'], predicted_svm),"%")

