#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib as plt


# In[2]:


w =  np.array([[1.], [2]])
b = 1.5
X = np.array([[1., -2., -1.], [3., 0.5, -3.2]])
Y = np.array([[1, 1, 0]])


# In[3]:


# Making Sigmoid function
def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s


# In[4]:


# intilisation of w and b
def w_b(X):
    w=np.zeros((X.shape[0],1))
    b=0.0
    return w,b


# In[5]:


# n=number of features
# m= number of data points
# w is vector of nx1 dimensions
# X should be nxm for the valid dot product of w.T and X
# Y is 1xm
m=X.shape[1]
def cost(w,b,X,Y):
    p=sigmoid(np.dot(w.T,X)+b)
    cost=-(np.dot(Y, np.log(p).T)+np.dot((1-Y), np.log(1-p).T))/m
    dw=np.dot(X,(p-Y).T)/m    # gradint of cost w.r.t w1,w2,w3,...etc
    db=np.sum(p-Y)/m          # gradient of cost w.r.t b which is w0
    
    costs = np.squeeze(np.array(cost))

    
    grads = {"dw": dw,
             "db": db}
    
    return grads, costs


# In[6]:


grads, costs=cost(w,b,X,Y)
print(cost)
print(grads["dw"])
print(grads["db"])


# In[7]:


dw = grads["dw"]
db = grads["db"]


# In[10]:


# I am finding the best tuned parameters using gradient dscent

def gradient_descent(w,b,X,Y,num_iterations=1000, l_r=0.07):

    
    for i in range(num_iterations):
        
        grads, costs = cost(w, b, X, Y)
        
       
        dw = grads["dw"]
        db = grads["db"]
        
        
        w=w-l_r*dw   # this is updation rul for 0,w1,w2....etc and l_r is learning rate 
        b=b-l_r*db   # this is updation rule for wo 
        

    
    parameters = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    print(i)
    return parameters


# In[11]:


parameters= gradient_descent(w,b,X,Y,num_iterations=1000, l_r=0.07)
parameters["w"]


# In[12]:


# now I am predicting the labels 
def predict(w, b, X):
    
    p=sigmoid(np.dot(w.T,X)+b)

    m = X.shape[1]
    Y_predict = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)
    
    for i in range(p.shape[1]):
        
        
        if p[0,i]>0.5:
            Y_predict[0,i] =1
        else:
            Y_predict[0,i] =0
        
    
    return Y_predict


# In[13]:


predict(w, b, X)


# In[61]:


def Logistic_Regression(w,b,X_train,Y_train,X_test,Y_test,num_iterations=1000, l_r=0.07):
    w,b=w_b(X_train)
    parameters= gradient_descent(w,b,X_train,Y_train,num_iterations=1000, l_r=0.07)
    w=parameters["w"]
    b=parameters["b"]
    Y_predict_train=predict(w,b,X_train)
    Y_predict_test=predict(w,b,X_test)
    Acuracy_train=np.sum(Y_predict_train==Y_train)/(X_train.shape[1])
    Acuracy_test=np.sum(Y_predict_test==Y_test)/(X_test.shape[1])
    # making a dictionary which will give final output
    out_put={"w":w, "b":b, "Y_predict_train":Y_predict_train, "Y_predict_test":Y_predict_test,
            "Acuracy_train":Acuracy_train, "Acuracy_test":Acuracy_test}
    return out_put
    


# In[62]:


diabetes=pd.read_csv("diabetes.csv")
diabetes


# In[63]:


# we have 8 fatures and 768 data points so n=8 and m=768
X=diabetes.iloc[:,0:8].values
print(X.shape)
Y=diabetes.loc[:,["Outcome"]].values
print(Y.shape)


# In[64]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.30, random_state=42)


# In[65]:


#X_train=X_train.T
#print(X_train.shape)
# y_train=y_train.T
# print(y_train.shape)
# X_test=X_test.T
#print(X_test.shape)
# y_test=y_test.T
# print(y_test.shape)
# # #X_test.shape


# In[66]:


X_train=X_train.T
print(X_train.shape)
y_train=y_train.T
print(y_train.shape)
X_test=X_test.T
print(X_test.shape)
y_test=y_test.T
print(y_test.shape)
# #X_test.shape


# In[67]:


len(X_train.shape[1])


# In[68]:


out_put=Logistic_Regression(w,b,X_train,y_train,X_test,y_test,num_iterations=1000, l_r=0.07)
out_put["w"]


# In[69]:


out_put["Acuracy_train"]


# In[70]:


out_put["Acuracy_test"]


# In[73]:


X_train=X_train.T
y_train=y_train.T


# In[75]:


from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(random_state=0).fit(X_train, y_train)
clf.score(X_train, y_train)


# In[ ]:




