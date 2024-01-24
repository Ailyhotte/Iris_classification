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
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

model_dict = {'Decision Tree':DecisionTreeClassifier(random_state=5),
              'Random Forest':RandomForestClassifier(random_state=5),
              'AdaBoost':AdaBoostClassifier(random_state=5),
              'KNN':KNeighborsClassifier(),
              'SVC':SVC(random_state=5)}

#%%
from sklearn.model_selection import learning_curve

def evaluation(model):
    model.fit(X_train, y_train)    
    N, train_score, val_score = learning_curve(model, X_train, y_train,
                                              cv=4, train_sizes=np.linspace(0.1, 1, 10))  
    plt.figure()
    plt.plot(N, train_score.mean(axis=1), label='train score')
    plt.plot(N, val_score.mean(axis=1), label='validation score')
    plt.legend()
    
    print('Train score :', model.score(X_train, y_train))
    print('Test score :', model.score(X_test, y_test))
#%%
for name, model in model_dict.items():
    print(name)
    evaluation(model)
#%%
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

model = make_pipeline(StandardScaler(), SVC(random_state=5))

evaluation(model)
#%%
from sklearn.model_selection import GridSearchCV

param_grid = {
    'svc__C': [0.1, 1, 10, 100],
    'svc__gamma': [0.01, 0.1, 1, 10],
    'svc__kernel': ['rbf']
}

grid = GridSearchCV(model, param_grid)

grid.fit(X_train, y_train)
evaluation(grid.best_estimator_)
#%%
model_2 = KNeighborsClassifier()
param_grid = {'n_neighbors':range(1,40),
              'metric':['euclidean','minkowski','manhattan']}
grid = GridSearchCV(model_2, param_grid, cv=5)
grid.fit(X_train, y_train)
evaluation(grid.best_estimator_)
#%%
from sklearn.linear_model import SGDClassifier
model_3 = make_pipeline(StandardScaler(), SGDClassifier(random_state=5))
param_grid = {'sgdclassifier__loss':['hinge', 'log_loss', 'modified_huber', 'squared_hinge', 'perceptron', 'squared_error', 'huber', 'epsilon_insensitive', 'squared_epsilon_insensitive'],
              'sgdclassifier__penalty':['l1','l2']}
grid = GridSearchCV(model_3, param_grid)
evaluation(grid)



