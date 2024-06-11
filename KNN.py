# -*- coding: utf-8 -*-
"""
Created on Wed May 10 07:46:32 2023

@author: okokp
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import math
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('D:/Code_Thread/Dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
dataset = pd.read_csv(r"D:/Code_Thread/Dataset/ClaMP_Integrated-5184.csv")

#------------------------------------------------------------------------------------------------
#Summary
print('Total Shape :',dataset.shape)
dataset.head()

type_df = pd.DataFrame(dataset.dtypes).reset_index()
type_df.columns=['cols','type']
type_df[type_df['type']=='object']['cols'].unique()

#------------------------------------------------------------------------------------
print('Total unique values in "packer_type":',dataset['packer_type'].nunique())
#------------------------------------------------------------------------------------
#Extracting the required levels only, based on value counts. 
packer_unique_df = pd.DataFrame(dataset['packer_type'].value_counts()).reset_index()
packer_unique_df.columns = ['packer_type','unique_count']
catg = packer_unique_df[packer_unique_df['unique_count']>10]['packer_type'].unique()
#------------------------------------------------------------------------------------
encoded = pd.get_dummies(dataset['packer_type'])
encoded = encoded[[col for col in list(encoded.columns) if col in catg]]
print('Shape of encode :',encoded.shape)
#------------------------------------------------------------------------------------
#Concatenating the encoded columns
if set(catg).issubset(set(dataset.columns))==False: #Conditional automation 
    dataset = pd.concat([dataset,encoded],axis=1)
    dataset.drop(columns='packer_type',inplace=True)

dataset.shape


from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Test Train Split for modelling purpose
X = dataset.loc[:,[cols for cols in dataset.columns if ('class' not in cols)]] #Removing time since its a level column
y = dataset.loc[:,[cols for cols in dataset.columns if 'class' in cols]]

#----------------------------------------------------------------------------------------------------
#Scaling the features
scaler = StandardScaler()
X = scaler.fit_transform(X)

#Splitting data into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33,random_state=100)

#----------------------------------------------------------------------------------------------------
print('Total Shape of Train X:',X_train.shape)
print('Total Shape of Train Y:',y_train.shape)
print('Total Shape of Test X:',X_test.shape)

#----------------------------------------------------------------------------------------------------

X_arr = np.array(X_train)
X_test_arr = np.array(X_test)

y_arr = np.array(y_train).reshape(len(y_train),1)
y_test_arr = np.array(y_test).reshape(len(y_test),1)

#----------------------------------------------------------------------------------------------------
print(X_arr.shape)
print(X_test_arr.shape)
print(y_arr.shape)

#distance calculation udf
def minkowski_(point_a,point_b,p=2):
    
    if p==1:
        print('----> Manhattan')
        dist = np.sum(abs(point_a-point_b))
        print('Manual Distance :',dist)
    elif p==2:
        #print('----> Euclidean')
        dist = np.sqrt(np.sum(np.square(point_a-point_b)))
        #print('Manual Distance :',dist)
        
    return dist

#------------------------------------------------------------------
#Calculate distance from one point to all other points including itself
def distance_to_all(curr_vec,data,p_=2):

    distance_list = []

    for vec_idx in range(len(data)):
        dist = minkowski_(point_a=curr_vec,point_b=data[vec_idx],p=p_)
        distance_list.append(dist)

    return distance_list


predictions = []
probabilities = []

def knn_model(data_x=X_arr,data_y=y_arr,k=10,curr_vec_=X_test_arr[34],mode='predict',threshold=0.5):

    #print('#--------------------------------------------------------------------------------')
    #Calculating distance of that point to every other point
    distance_list = distance_to_all(curr_vec=curr_vec_,data=data_x,p_=2)
    distance_list_reshaped = np.array(distance_list).reshape(len(distance_list),1)

    #print('#--------------------------------------------------------------------------------')
    #Creating a unified array for ease of indexing
    array_final = data_x
    array_final = np.append(array_final,data_y,axis=-1)
    array_final = np.append(array_final,distance_list_reshaped,axis=-1) #Appending distances
    
    #Sorting the datapoints by the distance column
    array_final_argsorted = array_final[array_final[:, -1].argsort()]

    if mode=='train':

        array_final_argsorted_top_k = array_final_argsorted[1:k+1,-2] #k+1 as the minimum distance is always 0 (with itself)
        ratio_ = np.sum(array_final_argsorted_top_k)/k #Total density around the point

        if ratio_>threshold:
            predictions.append(1)
        else:
            predictions.append(0)
            
    elif mode=='predict':

        array_final_argsorted_top_k = array_final_argsorted[0:k,-2] #Not k+1 since test data is not present in the training data (0 dist doesnt occur)
        ratio_ = np.sum(array_final_argsorted_top_k)/k

        if ratio_>threshold:
            pred = 1
        else:
            pred = 0

    return pred,ratio_ 

predictions = [] #Initializing predictions tray for each test datapoint
probabilities = [] #Initializing prediction probability tray for each test datapoint

for idx in range(len(X_test)): #Iterating for datapoint in test data
    #print('#-------------- ',idx,' --------------#')
    pred,prob = knn_model(data_x=X_arr,data_y=y_arr,
                          k=5,curr_vec_=X_test_arr[idx],
                          mode='predict',threshold=0.5)
    
    predictions.append(pred) #Appending into the tray
    probabilities.append(prob) #Appending into the tray
    
    
#-----------------------------------------------
#Evaluating the predictions from the KNN model
score = roc_auc_score(y_test_arr, predictions)
print('1. ROC AUC: %.3f' % score)
print('2. Accuracy :',accuracy_score(y_test_arr, predictions))
print('3. Classification Report -\n',classification_report(y_test_arr, predictions))
print('4. Confusion Matrix - \n',confusion_matrix(y_test_arr, predictions))

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_arr,y_arr)
sklearn_preds = knn.predict(X_test_arr)

#------------------------------------------------------------------------------------
score = roc_auc_score(y_test_arr, sklearn_preds)
print('1. ROC AUC: %.3f' % score)
print('2. Accuracy :',accuracy_score(y_test_arr, sklearn_preds))
print('3. Classification Report -\n',classification_report(y_test_arr, sklearn_preds))
print('4. Confusion Matrix - \n',confusion_matrix(y_test_arr, sklearn_preds))