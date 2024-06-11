# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score, classification_report
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('D:/Code_Thread/Dataset'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

file = open('D:/Code_Thread/Dataset/ClaMP_Integrated-5184.csv')
df = pd.read_csv(file, skip_blank_lines=True, na_filter=False, encoding='utf-8')

# We made Pandas DataFrame from data in ClaMP_Integrated-5184.csv file

pt = df['packer_type'].unique()
p_types = {pt[i] : i for i in range(len(pt))}
temp = []
for t in df['packer_type']:
    temp.append(p_types[t])
df['pt_num'] = temp
cl = df.pop('class')
df.pop('packer_type')

# 'packer_type' column changed to 'pt_num' column with corresponding integers

x_train, x_test, y_train, y_test = train_test_split(
    df, cl, random_state=0)

# DataFrame was splitted into training and testing sets

pipeStd = Pipeline([('scaler', StandardScaler()), ('svm', SVC(random_state=0))])
pipeStd.fit(x_train, y_train)

# We made a pipeline to properly scale and classify data
Pipeline(steps=[('scaler', StandardScaler()), ('svm', SVC(random_state=0))])
param_grid = {'svm__C':[0.1, 1, 10, 100, 200, 300],
    'svm__gamma':[0.001, 0.005, 0.01, 0.1, 1]}

grid = GridSearchCV(pipeStd, param_grid,
    cv = 5, n_jobs = -1)
grid.fit(x_train.to_numpy(), y_train)

# We found the best parameters for SVM Classifier and cross-validated them using GridSearchCV with pipeline of StandardScaler and SVC
GridSearchCV(cv=5,
             estimator=Pipeline(steps=[('scaler', StandardScaler()),
                                       ('svm', SVC(random_state=0))]),
             n_jobs=-1,
             param_grid={'svm__C': [0.1, 1, 10, 100, 200, 300],
                         'svm__gamma': [0.001, 0.005, 0.01, 0.1, 1]})
print('SVC score after StdScaler: {:.3f}'.format(
    grid.score(x_test.to_numpy(), y_test)))
print("SVC's best score on cross validation: {:.3f}".format(
    grid.best_score_))
print("Classifier's best parameters: {}".format(grid.best_params_))
pred_val = grid.predict(x_test.to_numpy())
print(classification_report(
    y_test, pred_val, target_names=['benign', 'malicious'], digits=3))

# Some core classification metrics

fpr, tpr, thresholds = roc_curve(
    y_test, grid.best_estimator_['svm'].decision_function(
        grid.best_estimator_['scaler'].transform(
            x_test.to_numpy())))
auc = roc_auc_score(y_test, grid.best_estimator_['svm'].decision_function(
        grid.best_estimator_['scaler'].transform(
            x_test.to_numpy())))
close_zero = np.argmin(np.abs(thresholds))

plt.figure(figsize=(5, 5), dpi=200)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')

plt.plot(fpr, tpr, label='ROC curve (AUC = {:.3f})'.format(auc), 
    color='g')
plt.plot(fpr[close_zero], tpr[close_zero], 'o', markersize=10,
    label="Absolute zero edge", fillstyle='none', color='r')
plt.legend(loc='lower right')
plt.title('SVC with StdScaler')

# ROC-AUC score with plot