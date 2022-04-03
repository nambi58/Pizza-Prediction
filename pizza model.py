#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
import pickle


# In[4]:


df1 = pd.read_csv("pizza.csv")
X = df1.drop(['likePizza'], axis=1)
y = df1['likePizza']


# In[5]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
knn.predict(X)


# In[6]:


pickle.dump((knn), open('model1.pkl','wb'))
model1 = pickle.load(open('model1.pkl','rb'))


# In[ ]:




