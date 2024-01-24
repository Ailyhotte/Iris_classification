#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 20:44:50 2024

@author: elliot
"""

"""
Created on Wed Jan 24 16:38:54 2024

@author: elliot
"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_dataset():
    data = sns.load_dataset('iris')
    return data

#%% 
data = load_dataset()
df = data.copy()
data.head()
#%%
sns.pairplot(df, hue='species')

#%%
data.isna().value_counts()
#no missing values
#%%
from sklearn.model_selection import train_test_split

y = df['species']
X = df.drop(['species'], axis=1)

df.dtypes.value_counts()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=5)
#%%
def evaluation(model):
    model.fit(X_train, y_train)    
    print('Train score :',model.score(X_train, y_train))
    print('Test score :',model.score(X_test, y_test))
#%%
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier()
evaluation(model)
#%%
#Résultat de 98% accuracy sur le train set et 93% d'accuracy sur le test set


