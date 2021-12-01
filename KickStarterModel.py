#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[2]:


dataset = pd.read_excel("KickstarterData_Facts-1 (1).xlsx")
dataset


# In[3]:


dataset.describe(include="all",datetime_is_numeric=True)


# In[4]:


dataset


# In[5]:


dataset.info()


# In[6]:


dataset.isnull().sum() 


# In[7]:


dataset['Purchased'].value_counts(dropna = False).plot(kind='bar', title="Number of Customer Purchased", figsize=(8.5, 7))


# In[8]:


sns.barplot(x="Deposit Amount", y= "Do you own a Keurig", data= dataset)


# In[9]:


sns.barplot(x="Donated To Kick Starter Before", y='Deposit Amount', hue='Purchased', data = dataset)


# In[10]:


correlation = dataset.corr()

ax = sns.heatmap(correlation,vmin=-1,vmax=1,center=0,
    cmap=sns.diverging_palette(10, 110, n=100),
    square=True, annot=True)


# In[11]:


fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(15, 8))
index = 0
axs = axs.flatten()
for k,v in dataset.select_dtypes(include='number').items():
    sns.boxplot(y=k, data=dataset, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)


# In[12]:


for col in dataset.keys():
    print("# # # # #  ",col,"  # # # # #")
    display(dataset[[col]].value_counts())


# In[13]:


dataset.drop(dataset[dataset['Deposit Amount']<=1].index, inplace = True)


# In[14]:


dataset = dataset.drop(['Donate ID','Donate Date'], axis=1)


# In[15]:


dataset['Donated To Kick Starter Before'] = dataset['Donated To Kick Starter Before'].map({'yes':1,'no':0})
dataset['Do you own a Keurig'] = dataset['Do you own a Keurig'].map({'yes':1,'no':0})


# In[16]:


dataset=pd.get_dummies(dataset,columns = dataset.select_dtypes(include='object').columns.values)
dataset


# In[17]:


y = dataset['Purchased'] 
x = dataset.drop(['Purchased'], axis=1)


# In[18]:


xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=.4,random_state=5)


# In[24]:


x_log= sm.add_constant(x)
logisticModel = sm.Logit(y, x_log) 
logisticModel_fit = logisticModel.fit()
print(logisticModel_fit.summary())


# In[25]:


pval = pd.DataFrame(logisticModel_fit.pvalues, columns=['pval'])
coef = pd.DataFrame(logisticModel_fit.params, columns= ['coef'])
combine = pd.concat([pval, coef], axis = 1)
combine


# In[26]:


combine[(combine['pval'] < 0.05)==True]


# In[27]:


logReg = LogisticRegression()
model = logReg.fit(xTrain, yTrain)


# In[28]:


yLPred = model.predict(xTest)
print(confusion_matrix(yTest, model.predict(xTest)))


# In[29]:


print('Accuracy  | '+str(round(accuracy_score(yTest, yLPred)*100,2))+'%')
print('Precision | '+str(round(precision_score(yTest, yLPred)*100,2))+'%')
print('Recall    | '+str(round(recall_score(yTest, yLPred)*100,2))+'%')
print('MSE       | '+str(round(mean_squared_error(yTest, yLPred)*100,2))+'%')


# In[30]:


DecisionTree = DecisionTreeClassifier(max_depth=3)
DecisionTree.fit(xTrain,yTrain)


# In[31]:


DecisionTree.score(xTest, yTest)


# In[32]:


plt.figure(figsize=(60,15))
feature_names = x.columns.values.tolist()
target_names = ['0','1']

tree.plot_tree(DecisionTree, 
          feature_names = feature_names, 
          class_names = target_names, 
          filled = True, 
          rounded = True)
plt.savefig('DecisionTreeChocolate.png') 


# In[33]:


from sklearn.ensemble import RandomForestClassifier
Random_Forest = RandomForestClassifier(n_estimators=100, criterion= 'entropy', random_state = 0)


# In[34]:


Random_Forest.fit(xTrain, yTrain)


# In[35]:


Random_Forest.score(xTrain, yTrain)


# In[36]:


y_pred_RF = Random_Forest.predict(xTest)


# In[37]:


confusion_matrix(yTest, y_pred_RF)

