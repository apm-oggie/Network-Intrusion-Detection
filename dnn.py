# -*- coding: utf-8 -*-
"""
Created on Wed May 10 07:57:14 2023

@author: okokp
"""

import numpy as np
import pandas as pd
     

data = pd.read_csv('D:/Code_Thread/dataset.csv')
     

data.head()
data.describe()

data["classification"].value_counts()
     

data['classification'] = data.classification.map({'nonissue':0, 'issue':1})
data.head()

# Shuffle data
data = data.sample(frac=1).reset_index(drop=True)
     

import matplotlib.pyplot as plt
import seaborn as sns
     

corrMatrix = data.corr()
sns.heatmap(corrMatrix, annot=True)
plt.show()


X = data.drop(["hash","classification",'vm_truncate_count','shared_vm','exec_vm','nvcsw','maj_flt','utime'],axis=1)
Y = data["classification"]
     

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=1)
     

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
     

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


import tensorflow as tf
     

#Number of attributes
input_size = 27 

#Number of Outputs
output_size = 2 

# Use same hidden layer size for all hidden layers.
hidden_layer_size = 64
    
# define how the model will look like
model = tf.keras.Sequential([
    tf.keras.layers.Dense(hidden_layer_size, input_shape=(input_size,), activation='relu'), # 1st hidden layer
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 2nd hidden layer
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(hidden_layer_size, activation='relu'), # 3rd hidden layer
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(output_size, activation='softmax') # output layer
])

model.summary()


model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
     

# set the batch size
batch_size = 100

# set a maximum number of training epochs
max_epochs = 30
     
history = model.fit(x=x_train, y=y_train, batch_size=batch_size, epochs=max_epochs, verbose=1, validation_split=0.2)

# Visualize the result
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
sns.set_style("white")
plt.suptitle('Train history', size = 15)

ax1.plot(epochs, acc, "bo", label = "Training acc")
ax1.plot(epochs, val_acc, "b", label = "Validation acc")
ax1.set_title("Training and validation acc")
ax1.legend()

ax2.plot(epochs, loss, "bo", label = "Training loss", color = 'red')
ax2.plot(epochs, val_loss, "b", label = "Validation loss", color = 'red')
ax2.set_title("Training and validation loss")
ax2.legend()

plt.show()
     

test_loss, test_accuracy = model.evaluate(x_test, y_test)
print('\nTest loss: {0:.6f}. Test accuracy: {1:.6f}%'.format(test_loss, test_accuracy*100.))
