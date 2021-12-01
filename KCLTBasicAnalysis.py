#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from pandas import Series
from matplotlib import pyplot


# In[2]:


dataset=pd.read_csv("KCLT.csv")
dataset


# In[3]:


def mov_mean(df,col,N):
    mov_avg = df[col].ewm(span=N).mean()
       
    return mov_avg

dataset['moving_avg'] = mov_mean(df=dataset,col ='actual_max_temp',N = 20)
dataset[['date','actual_max_temp','moving_avg']].head(40)


# In[4]:


plt.figure(figsize=(10,5))
pyplot.plot("date","actual_mean_temp",data=dataset,color="blue")
pyplot.plot("date","record_min_temp",data=dataset,color="#add8e6")
plt.legend(loc ='lower right')


# ###  - In the above chart you can see that real mean temp is parallel to the record min temperature for the dates from  2014 to 2015 so we can say that a similar example will be noticed for the following year 2016

# In[5]:


x=dataset["actual_mean_temp"].ewm(span=20).mean()
dataset["ewm_actual_mean_temp"]=x
plt.figure(figsize=(10,5))
pyplot.plot("date","record_max_temp",data=dataset,color="blue")
pyplot.plot("date","ewm_actual_mean_temp",data=dataset,color="#add8e6")
pyplot.plot("date","actual_max_temp",data=dataset,color="grey")
plt.legend(loc = 'lower right')


# ### -  In the second chart we can see that the temperature that was anticipated as max temperature was really higher than the actual max temperature so we can't rely on the recorded max temperature worth to foresee the real max temp for future years
