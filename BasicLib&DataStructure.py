#!/usr/bin/env python
# coding: utf-8

# In[14]:


#Import necessary libraries
import pandas as pd
import numpy as np


# In[15]:



def add_numbers(a, b):
    result = a + b
    return result

def sub_numbers(a, b):
    # Write code to return sum of a and b
    result =  a - b #Replace None with your code
    return result

def mult_numbers(a, b):
    #Code for mutiplication of a and b
    result = a * b # stored result of multiplication of a & b
    return result #return the result

print(mult_numbers(add_numbers(5,4), sub_numbers(5,4)))


# In[16]:



def print_name(name):
    formatted_result = "Your name is "+name # Added formated string with input name
    return formatted_result # return the formated statment

print_name("Josephine") # execute the procedure


# In[17]:


x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = {
    'Satya Nadella': 'Microsoft',
    'Jeff Bezos': 'Amazon',
    'Tim Cook': 'Apple'
}
def intDiv2Less6(a):
    result=[]
    for x in a: #For loop to iterate the list x
        if(x%2==0 and x < 6): #checked for conditions
            result.append(x)
    return result;

def strPrintList(y):
    for key,value in y.items(): # Iterate the list y as key value pair by using items()
        print("CEO: "+value+", Name: "+key) # Printed the CEO and Name on the screen

       
print('Items divisible by 2 :'+str(intDiv2Less6(x)))
strPrintList(y)


# In[ ]:




