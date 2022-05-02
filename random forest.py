#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import csv

train = pd.read_csv('./data/train.csv', index_col=0)
train = train.dropna(axis="index")
test_X = pd.read_csv('./data/test.csv', index_col=0)
test_y = pd.read_csv('./data/sample_submission.csv', index_col=0)

train_X = train.iloc[:, :-1]
train_y = train[train.columns[-1]]


# In[2]:


# Training and testing Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators = 100, min_samples_leaf = 25, random_state=18)
rf_model.fit(train_X, train_y)

preds = rf_model.predict(test_X)
print(f"Accuracy on train data: {accuracy_score(train_y, rf_model.predict(train_X))*100}")
 
print(f"Accuracy on test data: {accuracy_score(test_y, preds)*100}")


# In[3]:


predictions = rf_model.predict(test_X)

header = ['id', 'label']

with open('submission.csv', 'w', encoding='UTF8') as f:
    writer = csv.writer(f)

    writer.writerow(header)    
    for i in range(0, len(predictions)):
        writer.writerow([i,predictions[i]])


# In[ ]:





# In[ ]:




