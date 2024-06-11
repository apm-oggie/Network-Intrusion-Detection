# -*- coding: utf-8 -*-
"""
Created on Wed May 10 08:11:12 2023

@author: okokp
"""

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (10.0, 8.0)
import seaborn as sns

from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
Hardware= pd.read_excel("D:/Code_Thread/worm.xlsx")
Hardware.head()

Hardware.isnull().any()
Hardware['instructions']=Hardware['instructions'].fillna(Hardware['instructions'].mean())
Hardware.describe()

#bar plot for independent vs target variable
for i in Hardware.columns:
    if i!='class':
        fig = plt.figure(figsize = (10,6))
        sns.barplot(x = 'class', y = i, data = Hardware)
        
#Distribution of target variable
sns.countplot(Hardware['class'])

from sklearn.preprocessing import LabelEncoder
le_color = LabelEncoder()
Hardware['class'] = le_color.fit_transform(Hardware['class'])
Hardware['class'].unique()

#correlation plot
numeric_data = Hardware.select_dtypes(include=[np.number])
corr = numeric_data.corr()
sns.heatmap(corr)

plt.figure(figsize=(12,6))
sns.heatmap(Hardware.corr(),annot=True)


#Removal of outliers by interquartile range method
Q1 = Hardware.quantile(0.25)
Q3 = Hardware.quantile(0.75)
IQR = Q3 - Q1
Hardware=Hardware[~((Hardware < (Q1 - 1.5 * IQR)) |(Hardware > (Q3 + 1.5 * IQR))).any(axis=1)]
Hardware.shape


#removing multicollinearity by variance inflation factor method

y = Hardware['class']
Hardware = Hardware.drop(columns = ['class'])
Hardware.head()


from statsmodels.stats.outliers_influence import variance_inflation_factor
[variance_inflation_factor(Hardware.values, j) for j in range(Hardware.shape[1])]

def calculate_vif(x):
    thresh = 5.0
    output = pd.DataFrame()
    k = x.shape[1]
    vif = [variance_inflation_factor(x.values, j) for j in range(x.shape[1])]
    for i in range(1,k):        
        a = np.argmax(vif)        
        if vif[a] <= thresh :
            break
        if i == 1 :          
            output = x.drop(x.columns[a], axis = 1)
            #print(output)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
        elif i > 1 :
            output = output.drop(output.columns[a],axis = 1)
            #print(output)
            vif = [variance_inflation_factor(output.values, j) for j in range(output.shape[1])]
            
    return(output)
    
train_out = calculate_vif(Hardware) 
Hardware = Hardware.drop(columns=train_out.columns, axis=1)
Hardware.info()
#train_out.info()  
Hardware['class'] = y



print(Hardware.info())


# Separating out the features
X = Hardware.iloc[:, :-1].values  

# Separating out the target
y = Hardware.iloc[:,-1].values

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Standardizing the features
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)


