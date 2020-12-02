#!/usr/bin/env python
# coding: utf-8

#  # importing all the required libraries

# In[19]:



import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.linear_model import LogisticRegression


# ### coverting the given data into dataframe

# In[2]:


url = "http://bit.ly/w-data"
df = pd.read_csv(url)
df.head(10)


# #### correlation  between dependent anf idependent variables

# In[4]:


#we have good correlation hence we can use a regression model
df.corr()


# #### plotting a scatter plot 

# In[9]:


# Plotting the distribution of scores
plt.scatter(df['Hours'], df['Scores']) 
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# ## seperating dependent and idependent variables

# In[16]:


x = df.iloc[:,0].values 
y = df.iloc[:,1].values 


# ### splitting the data into training and testing data 

# In[21]:


from sklearn.model_selection import train_test_split  
X_train, X_test, y_train, y_test = train_test_split(x, y,test_size=0.2, random_state=0)


# #### using Logistic regression model

# In[29]:


model = LogisticRegression()
model.fit(X_train.reshape(-1,1),y_train.reshape(-1,1))


# ### predicted values using logistic regression

# In[36]:


y_pred = model.predict(X_test.reshape(-1,1))
y_pred


# ### # Comparing Actual vs Predicted

# In[33]:



df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df 


# In[44]:


from sklearn.metrics import r2_score
sklearn.metrics.r2_score(y_test,y_pred)


# In[46]:


from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




