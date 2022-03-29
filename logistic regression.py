#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

from sklearn.metrics import accuracy_score

train = pd.read_csv('./data/train.csv', index_col=0)
train = train.dropna(axis="index")

test_X = pd.read_csv('./data/test.csv', index_col=0)
test_y = pd.read_csv('./data/sample_submission.csv', index_col=0)

train_X = train.iloc[:, :-1]
train_y = train[train.columns[-1]]


# In[12]:


grid = {"C":np.logspace(-5, 5, 20), 
        "penalty":["l2"], 
        "solver":['newton-cg', 'lbfgs', 'saga',"liblinear"],
        "max_iter":[5000],
        "tol":[1e-3]
       }

model=GridSearchCV(LogisticRegression(), grid, cv=10)
model.fit(train_X, train_y)


# In[13]:


predictions = model.predict(test_X)
print("Logistics Regression accuracy: ", accuracy_score(predictions, test_y))
print()
print("Best parameters: ",model.best_params_)
print("Mean CV Accuracy: ",model.best_score_)


# In[37]:


import csv

header = ['id', 'label']

with open('submission.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)    
    for i in range(0, len(predictions)):
        writer.writerow([i,predictions[i]])

