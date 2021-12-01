#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import seaborn as sns


# In[36]:


dataset = pd.read_excel("KickstarterData.xlsx")
dataset


# In[37]:


dataset.shape


# In[38]:


dataset.describe(include="all",datetime_is_numeric=True)


# In[39]:


dataset.info()


# In[40]:


dataset.isnull().sum()


# In[41]:


dataset.shape


# In[42]:


sns.barplot(x='Deposit Amount',y ='Gender',data=dataset)


# In[43]:


sns.barplot(x='Household Income',y ='Ice Cream Products Consumed Per Week',data=dataset)


# In[44]:


sns.barplot(x="Donated To Kick Starter Before", y='Deposit Amount', hue='Gender', data = dataset)


# In[45]:


sns.barplot(x='Household Income', y='Deposit Amount', hue='Gender', data = dataset)


# In[46]:


dataset['How many desserts do you eat a week'].value_counts(dropna = False).plot(kind='bar', title="Number of Trees in each Borough", figsize=(8.5, 7))
ax.set_xlabel("Household Income")
ax.set_ylabel("Number of Trees")


# In[47]:


dataset['Preferred Color of Device'].value_counts(dropna = False).plot(kind='bar', title="Number of Trees in each Borough", figsize=(8.5, 7))
ax.set_xlabel("Household Income")
ax.set_ylabel("Number of Trees")


# In[48]:


dataset['Favorite Flavor Of Ice Cream'].value_counts(dropna = False).plot(kind='bar', title="Number of Trees in each Borough", figsize=(8.5, 7))
ax.set_xlabel("Household Income")
ax.set_ylabel("Number of Trees")


# In[49]:


dataset['Do you own a Keurig'].value_counts(dropna = False).plot(kind='bar', title="Number of Trees in each Borough", figsize=(8.5, 7))
ax.set_xlabel("Household Income")
ax.set_ylabel("Number of Trees")


# In[50]:


sns.barplot(x="Deposit Amount", y= "Do you own a Keurig", data= dataset)


# In[51]:


correlation = dataset.corr()

ax = sns.heatmap(correlation,vmin=-1,vmax=1,center=0,
    cmap=sns.diverging_palette(10, 110, n=100),
    square=True)


# In[52]:


dataset.skew()


# In[53]:


dataset['Deposit Amount'].fillna(100, inplace = True)
dataset.describe(include="all",datetime_is_numeric=True)


# In[54]:


dataset.isnull().sum() 


# In[55]:


dataset.skew()


# In[56]:


np.unique(dataset['Ice Cream Products Consumed Per Week'])


# In[57]:


dataset['Ice Cream Products Consumed Per Week'].fillna(0, inplace = True)


# In[58]:


dataset.skew()
dataset['Household Income'].value_counts(dropna = False)


# In[59]:


dataset['Household Income'].value_counts(dropna = False).plot(kind='bar', title="Number of Trees in each Borough", figsize=(8.5, 7))
ax.set_xlabel("Household Income")
ax.set_ylabel("Number of Trees")
ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x(), p.get_height()))


# In[60]:


dataset['Household Income'].fillna("Not Specified", inplace = True)


# In[61]:


sns.barplot(x='Household Income', y='Deposit Amount', hue='Donated To Kick Starter Before', data = dataset)


# In[62]:


sns.boxplot(x='How many desserts do you eat a week', data = dataset)


# In[63]:


dataset['How many desserts do you eat a week'].loc[dataset['How many desserts do you eat a week'] == 100] = 10


# In[64]:


sns.boxplot(x='Deposit Amount', data = dataset)

