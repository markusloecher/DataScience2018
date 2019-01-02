# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 16:52:38 2018

@author: ar810508
"""

import pandas as pd

import numpy as np
from pyearth import Earth
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy.stats import skew, skewtest

train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

train['SalePrice'].describe()

# string label to categorical values
from sklearn.preprocessing import LabelEncoder
for i in range(train.shape[1]):
    if train.iloc[:,i].dtypes == object:
        lbl = LabelEncoder()
        lbl.fit(list(train.iloc[:,i].values) + list(test.iloc[:,i].values))
        train.iloc[:,i] = lbl.transform(list(train.iloc[:,i].values))
        test.iloc[:,i] = lbl.transform(list(test.iloc[:,i].values))
        
# keep ID for submission
train_ID = train['Id']
test_ID = test['Id']

# split data for training
y_train = train['SalePrice']
X_train = train.drop(['Id','SalePrice'], axis=1)
X_test = test.drop('Id', axis=1)

for col in ('GarageYrBlt', 'GarageArea', 'GarageCars', 'LotFrontage', 'MasVnrArea','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath'):
    X_train[col] = X_train[col].fillna(0)
    X_test[col] = X_test[col].fillna(0)
        
#Missing = pd.concat([X_train.isnull().sum(), X_test.isnull().sum()], axis=1, keys=['train', 'test'])
#Missing[Missing.sum(axis=1) > 0]

X_train['TotalSF'] = X_train['TotalBsmtSF'] + X_train['1stFlrSF'] + X_train['2ndFlrSF']
X_test['TotalSF'] = X_test['TotalBsmtSF'] + X_test['1stFlrSF'] + X_test['2ndFlrSF']        


#normality check for the target
ax = sns.distplot(y_train)
plt.show()

#log transform the dependent variable for normality
y_train = np.log(y_train)
ax = sns.distplot(y_train)
plt.show()


#mars solution
model = Earth()

model = Earth(max_degree=2, penalty=1.0, minspan_alpha = 0.01, endspan_alpha = 0.01, endspan=5) #2nd degree formula is necessary to see interactions, penalty and alpha values for making model simple

model.fit(X_train, y_train)
model.score(X_train, y_train)
#y_pred = model.predict(train["SalePrice"])


y_pred = model.predict(X_test)
y_pred = np.exp(y_pred) # inverse log transform the results

print(model)
print(model.summary())
print(y_pred)

#Final_labels_new = np.expm1(model.predict(y_pred))

pd.DataFrame({'Id': test_ID, "SalePrice" : y_pred}).to_csv('Predictions1.csv', index =False)