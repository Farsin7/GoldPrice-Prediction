#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


# # Data collection and processing

# Basic information

# In[2]:


gold_data = pd.read_csv("gld_price_data.csv")
gold_data.head()


# In[8]:


gold_data.tail()


# In[5]:


gold_data.shape


# In[7]:


gold_data.info()


# In[9]:


gold_data.isnull().sum()


# In[11]:


gold_data.describe()


# Correlation
# 1. +ve 
# 2 - ve

# In[12]:


correlation = gold_data.corr()


# In[15]:


plt.figure(figsize=(6,6))
sns.heatmap(correlation, cbar=True, square=True, fmt=".1f", annot=True, annot_kws={'size':8}, cmap='Blues')


# In[18]:


# Correlation value of GLD

print(correlation['GLD'])


# In[21]:


#Distribution of GLD price

sns.displot(gold_data['GLD'], color='blue')


# # Splitting the Features and Targets

# In[24]:


X = gold_data.drop(['Date','GLD'], axis=1)
Y = gold_data['GLD']
print(Y)


# # Splitting into Training and Test Data

# In[26]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)


# Model Training: Random Forest Regressor
# 

# In[27]:


regressor = RandomForestRegressor(n_estimators=100)


# In[28]:


regressor.fit(X_train, Y_train)


# Model Evaluation

# In[29]:


test_data_prediction = regressor.predict(X_test)
print(test_data_prediction)


# R squared Error

# In[30]:


error_score = metrics.r2_score(Y_test, test_data_prediction)
print("R Squared Error:", error_score)


# In[ ]:


#Comparing actual vs predicted values


# In[31]:


Y_test = list(Y_test)


# In[33]:


plt.plot(Y_test, color='red', label='Actual')
plt.plot(test_data_prediction, color= 'yellow', label= 'Predicted')
plt.title('Actual VS Predicted Price')
plt.xlabel('No. of Values')
plt.ylabel('Gold Prices')
plt.legend()
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




