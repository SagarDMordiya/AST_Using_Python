#!/usr/bin/env python
# coding: utf-8

# In[1]:


#pip install numpy
#pip install pandas
#pip install urllib
#pip install seaborn
#pip install matplotlib
#pip install scipy
#pip install scikit-learn
#pip install statsmodels


# In[2]:


import numpy as np
import pandas as pd
import urllib.request
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


# Installed and Imported the packages needed for the smooth execution

# ## Introduction
# 
# In this assignment, we are going to perform the analysis on the dataset which contains the information about the crime incidents by Boston Police Department (BPD). We are going to perform the basic analysis on dataset which includes primilanary analysis, Data visualization. After that we are going to answer the bussiness question using the regression model. Lastly, we will address how we can improve or reduce the crime rate.

# ## Data Extraction
# 
# For this Assignment we have used the Dataset of Crime incident Reports provided by the Boston Police Department (BPD) and Dataset is publicly available on the website https://data.boston.gov/dataset/. For this assignment we are going to use the dataset is about the year 2021 and we are going to download the dataset using python and store the dataset for the further execution.
# 
# Direct Link for the Dataset :- https://data.boston.gov/dataset/6220d948-eae2-4e4b-8723-2dc8e67722a3/resource/f4495ee9-c42c-4019-82c1-d067f07e45d2/download/tmpsq1_rla4.csv
# 
# ### Bussines Question:
# 
# 1. Which offense has the highest crime rate?
# 2. How we can reduce the shooting rate?
# 3. Which District is unsafe?
# 4. Which months are safest and which are not?

# In[3]:


#Retriving data from Internet

#Dataset link
url = 'https://data.boston.gov/dataset/6220d948-eae2-4e4b-8723-2dc8e67722a3/resource/f4495ee9-c42c-4019-82c1-d067f07e45d2/download/tmpsq1_rla4.csv'

#retriving the dataset in current directory
urllib.request.urlretrieve(url, 'Crime_incident_Dataset.csv')


# Now, we have sucessfully downloaded the csv file from the website and store in the current directory to access later.
# After this we are going to read csv file and will start with the analysis.

# In[4]:


#reading downloaded dataset file
df = pd.read_csv('Crime_incident_Dataset.csv')

df


# ## Dataset Describtion

# In[5]:


df.info()


# In[6]:


df.describe(include = 'all')


# In[7]:


df.shape


# In[8]:


df.isnull().sum()


# In[9]:


df.isna().sum()


# In[10]:


for col in df:
    print ('\n Unique values of column: - %s'%col)
    print (df[col].unique())


# ## Data Visualization

# In[11]:


a=sns.boxplot(x=df['MONTH'])


# Boxplot of the crime incidents based on the month represents that the most the crime incidents are happened between 3 to 8 month.

# In[12]:


df['OFFENSE_CODE'].value_counts().nlargest(50).plot(kind='bar',figsize=(10,5))
plt.title('Frequency of Offense Code')
plt.ylabel('Number of Offense')
plt.xlabel('Type of Offense')


# Above plot represents the types of offense and number of time it occured which shows that the 3115 is most recorded offense follwed by 3005 and 3831 which directly answer the bussiness question but we will also try to look at the same after cleaning the dataset

# In[13]:


corr_df = df.corr()
ax = sns.heatmap(corr_df,vmin=-1,vmax=1,center=0,cmap=sns.diverging_palette(10, 110, n=100),square=True)
plt.show()


# It shows the correlation between each columns of the dataset and how they are interelated.

# In[14]:


df['DISTRICT'].value_counts().plot(kind='bar', title="Number of crime in each District", figsize=(8.5, 7))
ax.set_xlabel("District")
ax.set_ylabel("Number of Crimes")   
plt.show()


# Above plot shows the crime reported in the District and we can say that B2, D4 are unsafe district based on the crime records.This will help us to answer the bussiness question but we will also try to look at the same after cleaning the dataset

# In[15]:


monthlyGrp = df.groupby(df['MONTH'])['INCIDENT_NUMBER'].agg('count')
monthlyGrp.plot(kind='barh')


# Bar plot is able to show the crime incident count in each month and we can say 4 to 8 months are most vulnerable.

# In[16]:


monthlyGrp.plot()


# ## Data Cleaning

# In[17]:


df.info()


# In[18]:


#Droppping Null columns and unused column
df = df.drop(['OFFENSE_CODE_GROUP','UCR_PART','INCIDENT_NUMBER','OCCURRED_ON_DATE','OFFENSE_DESCRIPTION','Location'], axis = 1)


# In Aboove steps, we have dropped the columns which contains the no data and other columns has no importanted data as it contains the unique of incidents and location which not imported for modeling as we have latitude and longitude in seperate columns as well.

# In[19]:


for col in df:
    print ('\n Unique values of column: - %s'%col)
    print (df[col].unique())


# In[20]:


df.isnull().sum()


# In[21]:


#Dropping the NULL values
df = df.dropna()
df = df.drop(df[df.REPORTING_AREA == ' '].index)
df['REPORTING_AREA'] = pd.to_numeric(df['REPORTING_AREA'])


# Dropping the rows which contains the blank data in REPORTING_AREA filed as this is imported for analysis and after that we are converting the datatype to numeric

# In[22]:


df.info()


# In[23]:


#df['DAY_OF_WEEK']=preprocessing.LabelEncoder().fit_transform( df ['DAY_OF_WEEK'] )
df['DAY_OF_WEEK'] = df['DAY_OF_WEEK'].map({'Monday':1, 'Tuesday':2, 'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7})


# Converted the Categorical into numeric.

# In[24]:


df.head()


# ## Descriptive Analysis

# In[25]:


sns.heatmap(df.corr(),cmap='BuPu',annot=True)
plt.show()


# In[26]:


df['DISTRICT'].value_counts().plot(kind='bar', title="Number of crime in each District", figsize=(8.5, 7))


# In[27]:


monthlyGrp = df.groupby(df['MONTH'])['DISTRICT'].agg('count')
monthlyGrp.plot(kind = 'bar')
plt.title('Frequency of Offense in Month')
plt.ylabel('Number of Offense')
plt.xlabel('Month')


# In[28]:


df['OFFENSE_CODE'].value_counts().nlargest(50).plot(kind='bar',figsize=(10,5))
plt.title('Frequency of Offense Code')
plt.ylabel('Number of Offense')
plt.xlabel('Type of Offense')


# ## Predictive analysis

# #### Preparing the dataset for the Logistic model

# In[29]:


df = pd.get_dummies(df, columns=['DISTRICT'])
df.info()


# Created the Dummy variables for the district

# In[30]:


y = df['SHOOTING']
x = df.drop(['SHOOTING','STREET'], axis = 1)


# Divided the dataset into two part as Shooting as Depenedent variable and Other parameter as Independent variables. Also we are going to divide the dataset into 7:3 ratio for train and test dataset

# In[31]:


xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=.3 ,random_state=9)


# ### How we can reduce the shooting rate?
# For this we need to know the independent variable affeced the dependent variable (Shooting) to reduce the shooting rate.

# In[32]:


xConstant = sm.add_constant(x)
lmodel = sm.Logit(y, xConstant) 
lmodelFit = lmodel.fit()
print(lmodelFit.summary())


# After exection of the model, we know that how independent variable affects the shooting. Let's reduce the result to know more.

# In[33]:


pVal = pd.DataFrame(lmodelFit.pvalues, columns=['pValue'])
coefVal = pd.DataFrame(lmodelFit.params, columns= ['coef'])
LReg = pd.concat([pVal, coefVal], axis = 1)
LReg[(LReg['pValue'] < 0.05)==True]


# Now, we have two most significant variable which are affected the shooting rate which we can use to reduce the rate.

# In[34]:


logisticReg = LogisticRegression()
model = logisticReg.fit(xTrain, yTrain)


# In[35]:


yLPred = logisticReg.predict(xTest)
print(confusion_matrix(yTest, model.predict(xTest)))


# In[36]:


print("Classification report - \n", classification_report(yTest,yLPred))


# Accuracy of the model is 99% which indicates that the logisitic model performed well and we can make use of this to reduce the shooting incidents.

# ## Conclusion
# 
# In the conclusion of the this assignment, we have utilised the various method and techniques we learned throughout the course to solve this assignment. We have downloaded the publicly available dataset and performed the exploriatory data analysis and created viuslization. Apart from that after data cleaning, we have created some Descriptive and Predictive analysis for that we have used the logistic model to know how we can reduce the crime rate and overall it was difficult to perform the analysis but I enjoyed the whole process of assignment.  
# 

# # Reference
# ### 
# Robinson, S. (2017, October 31). Download Files with Python. Stack Abuse. https://stackabuse.com/download-files-with-python/
# 
# Gallagher, J. (2021, January 20). Python Average: A Step-by-Step Guide. Career Karma. https://careerkarma.com/blog/python-average/
# 
# sklearn.preprocessing.LabelEncoder — scikit-learn 0.24.2 documentation. (n.d.). Https://Scikit-Learn.Org/Stable/Modules/Generated/Sklearn.Preprocessing.LabelEncoder.Html
# 
