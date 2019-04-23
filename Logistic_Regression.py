#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


#import dataset using pandas
adult_data = pd.read_csv("C:/Users/hp/Documents/Northeastern University/Predictive_Analytics_ALY6020/Week_5/Assignment/adult.csv")
adult_data.head()


# In[3]:


#Preparing the data set
adult_all = list(adult_data.shape)[0]
adult_categories = list(adult_data['ABOVE50K'].value_counts())

print("The dataset has {} count , {} below 50k and {} above 50k.".format(adult_all, 
                                                                                 adult_categories[0], 
                                                                                 adult_categories[1]))


# In[6]:


#converting non-numeric values to numeric
adult_data['WORKCLASS'],_ = pd.factorize(adult_data['WORKCLASS'])  
adult_data['FNLWGT'],_ = pd.factorize(adult_data['FNLWGT'])  
adult_data['EDUCATION'],_ = pd.factorize(adult_data['EDUCATION'])  
adult_data['MARITALSTATUS'],_ = pd.factorize(adult_data['MARITALSTATUS'])  
adult_data['OCCUPATION'],_ = pd.factorize(adult_data['OCCUPATION']) 
adult_data['RELATIONSHIP'],_ = pd.factorize(adult_data['RELATIONSHIP'])  
adult_data['RACE'],_ = pd.factorize(adult_data['RACE'])  
adult_data['SEX'],_ = pd.factorize(adult_data['SEX'])  
adult_data['NATIVECOUNTRY'],_ = pd.factorize(adult_data['NATIVECOUNTRY']) 
adult_data.head()


# In[7]:


adult_data.describe()


# In[8]:


#Normalizing numeric data
def normalize(dataset):
    dataNorm=((dataset-dataset.min())/(dataset.max()-dataset.min())) 
    dataNorm["ABOVE50K"]=dataset["ABOVE50K"]
    return dataNorm
adult_data=normalize(adult_data)
adult_data.describe()


# In[9]:


#Creating training and test datasets for Logistic Regression
from sklearn.model_selection import train_test_split

X = adult_data.drop(["ABOVE50K"], axis = 1)
y = adult_data.ABOVE50K

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 40)


# In[21]:


#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix 
from sklearn.metrics import accuracy_score

Regression_Logistic = LogisticRegression(C = 20.0, random_state=1, solver='newton-cg', max_iter=550, multi_class='multinomial')
Regression_Logistic.fit(X_train, y_train)

y_pred = Regression_Logistic.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))  
print('')
print('Classification Report:')
print(classification_report(y_test, y_pred)) 


# In[22]:


Regression_Logistic = LogisticRegression(class_weight = 'balanced',C = 20.0, random_state=1, 
                                         solver='newton-cg', max_iter=550, multi_class='multinomial')
Regression_Logistic.fit(X_train, y_train)

y_pred = Regression_Logistic.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))  
print('')
print('Classification Report:')
print(classification_report(y_test, y_pred)) 


# In[23]:


from sklearn.metrics import log_loss
log_loss(y_test, y_pred)


# In[ ]:




