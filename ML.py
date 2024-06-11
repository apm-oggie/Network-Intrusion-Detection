# -*- coding: utf-8 -*-
"""
Created on Wed May 10 07:54:10 2023

@author: okokp
"""

import pandas as pd
import numpy as np
mal_data = malData = pd.read_csv('systemdata.csv',sep='|')
mal_data.head()
X = pd.DataFrame(mal_data.iloc[:,2:-1])
X
y = pd.DataFrame(mal_data.iloc[:,-1])
y
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=0)
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=20,criterion='gini',random_state=1,max_depth=3)
classifier.fit(X_train,y_train.values.ravel())

from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=0,solver="newton-cg") 
logModel = clf.fit(X_train, y_train.values.ravel())
from sklearn.ensemble import AdaBoostClassifier

AdaModel = AdaBoostClassifier(n_estimators=100,learning_rate=1)
AdaModel.fit(X_train, y_train.values.ravel())

from sklearn.naive_bayes import BernoulliNB
naivebayesmodel=BernoulliNB()
naivebayesmodel.fit(X_train, y_train.values.ravel())
from sklearn.neighbors import KNeighborsClassifier  
knn= KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2 ) 
knn.fit(X_train, y_train.values.ravel())

from sklearn.metrics import confusion_matrix
import seaborn as sns
knn_pred=knn.predict(X_test)
cf_matrix=confusion_matrix(y_test, knn_pred)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',fmt='g')
ax.set_title('Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(['Negative','Positive'])
ax.yaxis.set_ticklabels(['Negative','Positive'])

from sklearn.metrics import confusion_matrix
import seaborn as sns
forest_pred=classifier.predict(X_test)
cf_matrix=confusion_matrix(y_test, forest_pred)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',fmt='g')
ax.set_title('Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(['Negative','Positive'])
ax.yaxis.set_ticklabels(['Negative','Positive'])

logistic_pred=clf.predict(X_test)
cf_matrix=confusion_matrix(y_test, logistic_pred)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',fmt='g')
ax.set_title('Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(['Negative','Positive'])
ax.yaxis.set_ticklabels(['Negative','Positive'])

ada_pred=AdaModel.predict(X_test)
cf_matrix=confusion_matrix(y_test, ada_pred)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',fmt='g')
ax.set_title('Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(['Negative','Positive'])
ax.yaxis.set_ticklabels(['Negative','Positive'])

naivebayesmodel_pred=naivebayesmodel.predict(X_test)
cf_matrix=confusion_matrix(y_test, naivebayesmodel_pred)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues',fmt='g')
ax.set_title('Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');
ax.xaxis.set_ticklabels(['Negative','Positive'])
ax.yaxis.set_ticklabels(['Negative','Positive'])

from sklearn.metrics import precision_score, recall_score, f1_score,accuracy_score
print('Accuracy for Random Forest: %.3f' %accuracy_score(y_test, forest_pred))
print('Accuracy for Logistic Regression: %.3f' %accuracy_score(y_test, logistic_pred))
print('Accuracy for Adaboost: %.3f' %accuracy_score(y_test, ada_pred))
print('Accuracy for Naive Bayes Classifier: %.3f' %accuracy_score(y_test,naivebayesmodel_pred))
print('Accuracy for K Nearest Neighbour Classifier: %.3f' %accuracy_score(y_test,knn_pred))

print('F1 Score for Random Forest: %.3f' %f1_score(y_test, forest_pred))
print('F1 Score for Logistic Regression: %.3f' %f1_score(y_test, logistic_pred))
print('F1 Score for Adaboost: %.3f' %f1_score(y_test, ada_pred))
print('F1 Score for Naive Bayes Classifier: %.3f' %f1_score(y_test, naivebayesmodel_pred))
print('F1 Score for K Nearest Neighbour Classifier: %.3f' %f1_score(y_test,knn_pred))