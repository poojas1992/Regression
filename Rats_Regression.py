#!/usr/bin/env python
# coding: utf-8

# In[18]:


import pandas as pd 


# In[363]:


rats_data = pd.read_csv("C:/Users/hp/Documents/Northeastern University/Predictive_Analytics_ALY6020/Week_5/Assignment/rats.csv")
rats_data.head()


# In[364]:


#Preparing the data set
rats_all = list(rats_data.shape)[0]
rats_categories = list(rats_data['status'].value_counts())

print("The dataset has {} types , {} has no tumor and {} has tumor.".format(rats_all, 
                                                                                 rats_categories[0], 
                                                                                 rats_categories[1]))


# In[365]:


X = rats_data.drop(["status"], axis = 1)
y = rats_data.status 


# In[366]:


#Creating training and test datasets 
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state = 1)


# In[371]:


# Train the model using the training sets
from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix  

#Linear Regression
Rat_Regression = linear_model.LinearRegression()
Rat_Regression.fit(X_train, y_train)

y_pred = Rat_Regression.predict(X_test)

print('Coefficients: \n', Rat_Regression.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

y_pred_b=[1 if x>0.5 else 0 for x in y_pred]

accuracy = accuracy_score(y_test, y_pred_b)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

print(confusion_matrix(y_test, y_pred_b))  
print(classification_report(y_test, y_pred_b)) 

plt.scatter(y_test, y_pred,  color='black')

plt.xticks(())
plt.yticks(())

plt.show()


# In[368]:


# Train the model using the training sets
from sklearn import linear_model

#Ridge Regression
Rat_Regression = linear_model.Ridge()
Rat_Regression.fit(X_train, y_train)

y_pred = Rat_Regression.predict(X_test)

print('Coefficients: \n', Rat_Regression.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

y_pred_b=[1 if x>0.5 else 0 for x in y_pred]

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_b)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred_b))  
print(classification_report(y_test, y_pred_b)) 

plt.scatter(y_test, y_pred,  color='black')

plt.xticks(())
plt.yticks(())

plt.show()


# In[369]:


#Lasso Regression
Rat_Regression = linear_model.Lasso(alpha = 0.1)
Rat_Regression.fit(X_train, y_train)

y_pred = Rat_Regression.predict(X_test)

print('Coefficients: \n', Rat_Regression.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))

y_pred_b=[1 if x>0.5 else 0 for x in y_pred]

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred_b)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

from sklearn.metrics import classification_report, confusion_matrix  
print(confusion_matrix(y_test, y_pred_b))  
print(classification_report(y_test, y_pred_b)) 

plt.scatter(y_test, y_pred,  color='black')

plt.xticks(())
plt.yticks(())

plt.show()


# In[ ]:




