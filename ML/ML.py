# -*- coding: utf-8 -*-
"""
Created on Mon Apr 23 12:08:11 2018

@author: jgadasu1
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import r2_score

# Importing the dataset
dataset = pd.read_csv('Frame.csv')
dataset.drop(['Unnamed: 0'],axis = 1)
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 5].values


# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
y = labelencoder.fit_transform(y)

list_of_values = dataset.values.tolist()
positive = (dataset.iloc[:,1].values )
neutral = (dataset.iloc[:,2].values)
negative = (dataset.iloc[:,3].values)

#y.reshape(-1,1)
#onehotencoder = OneHotEncoder(categorical_features = [9])
#y = onehotencoder.fit_transform(y).toarray()


# Avoiding the Dummy Variable Trap
#X = X[:, 1:]


# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)


y_pred = regressor.predict(X_test)

y_pred = (list(y_pred))
y_test = (list(y_test))

print (r2_score(y_test,y_pred))