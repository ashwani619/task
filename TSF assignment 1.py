#!/usr/bin/env python
# coding: utf-8

# # The Spark Foundation #GRIPMAY2021

# ## Task 1 - Prection using supervised ML

# ### By:- Ashwani Gautam

# In[ ]:


# Importing all libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


# Reading the data
url='https://raw.githubusercontent.com/AdiPersonalWorks/Random/master/student_scores%20-%20student_scores.csv'


# In[3]:


df=pd.read_csv(url)


# In[4]:


df.head()


# In[5]:


# Plotting the relationship between hours and score
df.plot(x='Hours',y='Scores',style='x')
plt.title('hours vs percentage')
plt.xlabel('hours')
plt.ylabel('percentage')
plt.show()


# ### From the above graph,we can see there is a positive relationship between hours and score

# In[6]:


# Divide the data into input and output
x=df.iloc[:,0:1]
y=df.iloc[:,1:]


# ## Training the data

# In[8]:


from sklearn.model_selection import train_test_split


# In[9]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


# In[10]:


from sklearn.linear_model import LinearRegression


# In[11]:


lr=LinearRegression()


# In[12]:


lr.fit(x_train,y_train)


# In[13]:


lr.score(x_train,y_train)


# In[14]:


lr.score(x_test,y_test)


# In[15]:


pred=lr.predict(x_test)


# In[16]:


from sklearn.metrics import mean_squared_error,mean_absolute_error


# In[17]:


print(mean_squared_error(pred,y_test))


# In[18]:


print(np.sqrt(mean_squared_error(pred,y_test)))


# In[19]:


#plotting the best fit line
line = lr.coef_*x+lr.intercept_

plt.scatter(x,y)
plt.plot(x,line)
plt.show()


# ## Making the prediction

# In[20]:


df2=pd.DataFrame(y_test)
df2


# In[21]:


df2['prediction']=pred


# In[22]:


# Comparison between actual and predicted
df2


# In[23]:


#Test with your own data
hours= [[9.25]]


# In[24]:


pred2=lr.predict(hours)


# In[25]:


pred2


# # no. of hours is 9.25 and predicted score is 93.89
