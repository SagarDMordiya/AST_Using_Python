
#import pandas as pd
#import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import statsmodels.api as sm
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Data Extraction
dataset = pd.read_excel("KickstarterData_Facts-1 (1).xlsx")
dataset

# Data Exploration

dataset.describe(include="all",datetime_is_numeric=True)

dataset

dataset.info()

dataset.isnull().sum() 

# Data visualization

dataset['Purchased'].value_counts(dropna = False).plot(kind='bar', title="Number of Customer Purchased", figsize=(8.5, 7))

sns.barplot(x="Deposit Amount", y= "Do you own a Keurig", data= dataset)

sns.barplot(x="Donated To Kick Starter Before", y='Deposit Amount', hue='Purchased', data = dataset)

correlation = dataset.corr()

ax = sns.heatmap(correlation,vmin=-1,vmax=1,center=0,
    cmap=sns.diverging_palette(10, 110, n=100),
    square=True, annot=True)


fig, axs = plt.subplots(ncols=5, nrows=1, figsize=(15, 8))
index = 0
axs = axs.flatten()
for k,v in dataset.select_dtypes(include='number').items():
    sns.boxplot(y=k, data=dataset, ax=axs[index])
    index += 1
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=5.0)

# Data Cleaning

for col in dataset.keys():
    print("# # # # #  ",col,"  # # # # #")
    display(dataset[[col]].value_counts())

dataset.drop(dataset[dataset['Deposit Amount']<=1].index, inplace = True)

dataset = dataset.drop(['Donate ID','Donate Date'], axis=1)

dataset['Donated To Kick Starter Before'] = dataset['Donated To Kick Starter Before'].map({'yes':1,'no':0})
dataset['Do you own a Keurig'] = dataset['Do you own a Keurig'].map({'yes':1,'no':0})

# Data preparation

dataset=pd.get_dummies(dataset,columns = dataset.select_dtypes(include='object').columns.values)
dataset

y = dataset['Purchased'] 
x = dataset.drop(['Purchased'], axis=1)

xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=.4,random_state=5)

# Data Modeling

#Logistic regression 

x_log= sm.add_constant(x)
logisticModel = sm.Logit(y, x_log) 
logisticModel_fit = logisticModel.fit()
print(logisticModel_fit.summary())

pval = pd.DataFrame(logisticModel_fit.pvalues, columns=['pval'])
coef = pd.DataFrame(logisticModel_fit.params, columns= ['coef'])
combine = pd.concat([pval, coef], axis = 1)
combine

combine[(combine['pval'] < 0.05)==True]

logReg = LogisticRegression()
model = logReg.fit(xTrain, yTrain)

yLPred = model.predict(xTest)
print(confusion_matrix(yTest, model.predict(xTest)))

print('Accuracy  | '+str(round(accuracy_score(yTest, yLPred)*100,2))+'%')
print('Precision | '+str(round(precision_score(yTest, yLPred)*100,2))+'%')
print('Recall    | '+str(round(recall_score(yTest, yLPred)*100,2))+'%')
print('MSE       | '+str(round(mean_squared_error(yTest, yLPred)*100,2))+'%')

#Decision Tree Model

DecisionTree = DecisionTreeClassifier(max_depth=3)
DecisionTree.fit(xTrain,yTrain)

DecisionTree.score(xTest, yTest)

plt.figure(figsize=(60,15))
feature_names = x.columns.values.tolist()
target_names = ['0','1']

tree.plot_tree(DecisionTree, 
          feature_names = feature_names, 
          class_names = target_names, 
          filled = True, 
          rounded = True)
plt.savefig('DecisionTreeChocolate.png') 

#Random Forest Model

from sklearn.ensemble import RandomForestClassifier
Random_Forest = RandomForestClassifier(n_estimators=100, criterion= 'entropy', random_state = 0)

Random_Forest.fit(xTrain, yTrain)

Random_Forest.score(xTrain, yTrain)

y_pred_RF = Random_Forest.predict(xTest)

confusion_matrix(yTest, y_pred_RF)

