#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from flask import Flask, render_template,request
import pickle


# In[3]:


app = Flask(__name__)
model1 = pickle.load(open('model1.pkl', 'rb'))


# In[2]:


df = pd.read_csv('pizza.csv')
x=df.drop(['likePizza'],axis=1)
y=df['likePizza']


# In[4]:


@app.route('/')
def home():
    return render_template('index.html')


# In[5]:


@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    float_features = [int(x) for x in request.form.values()]
    final_features = [np.array(float_features)]
    prediction = model1.predict(final_features) 

    if prediction == 1:
        pred = "You like pizza."
    elif prediction == 0:
        pred = "You don't like pizza."
    output = pred

    return render_template('index.html', prediction_text='{}'.format(output))


# In[6]:


if __name__ == "__main__":
    app.run(debug=True)


# In[ ]:




