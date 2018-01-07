# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 20:20:29 2018

@author: Tonmoy
"""

import pandas as pd

#importing the dataset
dataset = pd.read_csv("Churn_Modelling.csv")
x = dataset.iloc[:,3:13].values
y = dataset.iloc[:,13].values

#encode categorical values
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x1 = LabelEncoder()
onehotencoder = OneHotEncoder(categorical_features = [1])
x[:,1] = labelencoder_x1.fit_transform(x[:,1])
labelencoder_x2 = LabelEncoder()
x[:,2] = labelencoder_x2.fit_transform(x[:,2])
x = onehotencoder.fit_transform(x).toarray()

#delete one row to avoid dummy variable trap
x = x[:,1:]

#split into test and train
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#ANN
import keras
from keras.models import Sequential
from keras.layers import Dense

#adding input layer and first hidden layer(basically first hidden)
#output dim now known as units = avg of input +out = 11+1 (trick for beginners, not recommended later), relu is an activation fn - rectifier fn
classifier = Sequential()
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu", input_dim=11))

#add second hiddden layer...this layer knows what input to expect so input_dim is not required
classifier.add(Dense(units=6, kernel_initializer="uniform", activation="relu"))

#output layer
#only one node in output layer,so output_dim or units =1
#incase of more than 2 category of possible outputs, use softmax inplace of sigmoid and units = 3,etc
classifier.add(Dense(units=1, kernel_initializer="uniform", activation="sigmoid", input_dim=11))

#compiling the ANN = apply stochastic gradient descent
#optimizer is the algo to minimize cost fn, loss is the cost fn (sum of squared error)
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

#fitting ANN
#batch size and epoch like output_dim is art, need tuning for accuracy
classifier.fit(x_train, y_train, batch_size = 10, epochs = 100)

#prediction
y_pred = classifier.predict(x_test)
y_pred = (y_pred > 0.5) #this is done for confusion matrix below

#confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
