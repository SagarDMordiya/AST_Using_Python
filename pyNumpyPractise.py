#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import necessary libraries
import pandas as pd
import numpy as np


# In[2]:


arr = np.arange(24).reshape(4, 6)
def funct1():
    try:
        print('Determine the shape, number of dimensions and type of elements in the numpy array arr:')
        print("Shape - "+str(arr.shape))
        print("Dimension - "+str(arr.ndim))
        print("Type of elements - "+str(arr.dtype)+'\n')
    except Exception as e:
        print("Exception occured - "+str(e))

def funct2():
    try:
        print('Determine the sum of elements in arr')
        print('Sum - '+str(arr.sum())+'\n')
    except Exception as e:
        print("Exception occured - "+str(e))

def funct3():
    try:
        print('Determine the mean of elements in arr')
        print('Mean - '+str(arr.mean())+'\n')
        return
    except Exception as e:
        print("Exception occured - "+str(e))

def funct4():
    try:
        print('Create an array of the same shape as arr but filled with zeros')
        print('New Array of Zeros - ')
        print(np.zeros_like(arr))
        print('\n')
    except Exception as e:
        print("Exception occured - "+str(e))

def funct5():
    try:
        print('Create an array of the same shape as arr but filled with ones')
        print('New Array of Ones - ')
        print(np.ones_like(arr))
        print('\n')
    except Exception as e:
        print("Exception occured - "+str(e))

def funct6():
    try:
        print('Create an array of the same shape as arr but where all elements are squared values')
        print('Squared array - ')
        print(np.full_like(arr, np.square(arr)))
        print('\n')
    except Exception as e:
        print("Exception occured - "+str(e))

def funct7():
    try:
        print('Create an array result of shape 4 by 4 resulting from multiplication of arr with transpose(arr)')
        print('Result array - ')
        print(np.dot(arr, np.transpose(arr)))
        print('\n')
    except Exception as e:
        print("Exception occured - "+str(e))
        
def default():
    return "Invalid option"

switcher = {
    0: funct1,
    1: funct2,
    2: funct3,
    3: funct4,
    4: funct5,
    5: funct6,
    6: funct7
    }

def switch(quesNo):
    return switcher.get(quesNo, default)()

print("Output of the functions:")
for x in range(7):
    switch(x)


# In[3]:



import pandas as pd
df = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
print(df) #prints the first 10 rows
def quest1():
    try:
        print('(a) Determine the shape of the dataframe')
        print("Shape - "+str(df.shape))
        print('\n')
    except Exception as e:
        print("Exception occured - "+str(e))

def quest2():
    try:
        print('(b) Print out all the columns of the dataframe')
        print('Columns - '+str(df.columns.values)+'\n')
    except Exception as e:
        print("Exception occured - "+str(e))

def quest3():
    try:
        print('(c) Print the 3rd element of the dataframe')
        print('3rd Element - ')
        print(df.iloc[2:3])
        print('\n')
        return
    except Exception as e:
        print("Exception occured - "+str(e))

def quest4():
    try:
        print('(d) Find the average of ''petal_width'' where species is ''virginica''')
        print('Average width of Virginica - ')
        print(round(df.loc[df['species']=='virginica']['petal_width'].mean(),2))
        print('\n')
    except Exception as e:
        print("Exception occured - "+str(e))

def quest5():
    try:
        print('(e) Find the maximum of ''sepal_width'' where species is ''setosa''')
        print('Maximum width of setosa - ')
        print(round(df.loc[df['species']=='setosa']['sepal_width'].max(),2))
        print('\n')
    except Exception as e:
        print("Exception occured - "+str(e))

def quest6():
    try:
        print('(f): What is the average value of sepal_length')
        print('Average of Sepal Length - ')
        print(round(df['sepal_length'].mean(),2))
        print('\n')
    except Exception as e:
        print("Exception occured - "+str(e))

def quest7():
    try:
        print('(g): What is the maximum value of sepal_width')
        print('maximum of sepal_width - ')
        print(round(df['sepal_width'].max(),2))
        print('\n')
    except Exception as e:
        print("Exception occured - "+str(e))

def quest8():
    try:
        print('(h): What is the minimum value of petal_width')
        print('minimum of petal_width - ')
        print(round(df['petal_width'].min(),2))
        print('\n')
    except Exception as e:
        print("Exception occured - "+str(e))
        
def default():
    return "Invalid option"

switcher = {
    0: quest1,
    1: quest2,
    2: quest3,
    3: quest4,
    4: quest5,
    5: quest6,
    6: quest7,
    7: quest8
    }

def switch(questNo):
    return switcher.get(questNo, default)()

print("Answers of the questions:\n")
for x in range(8):
    switch(x)

