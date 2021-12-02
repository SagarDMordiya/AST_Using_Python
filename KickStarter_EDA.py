
import pandas as pd
import numpy as np
import seaborn as sns

# Data Extraction

dataset = pd.read_excel("KickstarterData.xlsx")
dataset

# Data Exploration

dataset.shape

dataset.describe(include="all",datetime_is_numeric=True)

dataset.info()

dataset.isnull().sum()

dataset.shape

# Data Visulization

sns.barplot(x='Deposit Amount',y ='Gender',data=dataset)

sns.barplot(x='Household Income',y ='Ice Cream Products Consumed Per Week',data=dataset)

sns.barplot(x="Donated To Kick Starter Before", y='Deposit Amount', hue='Gender', data = dataset)

sns.barplot(x='Household Income', y='Deposit Amount', hue='Gender', data = dataset)

dataset['How many desserts do you eat a week'].value_counts(dropna = False).plot(kind='bar', title="Number of Desserts eat in Week", figsize=(8.5, 7))
ax.set_xlabel("Week")
ax.set_ylabel("Number of Desserts")

dataset['Preferred Color of Device'].value_counts(dropna = False).plot(kind='bar', title="Preferred Color of Device", figsize=(8.5, 7))

dataset['Favorite Flavor Of Ice Cream'].value_counts(dropna = False).plot(kind='bar', title="Number of Trees in each Borough", figsize=(8.5, 7))

dataset['Do you own a Keurig'].value_counts(dropna = False).plot(kind='bar', title="Number of Trees in each Borough", figsize=(8.5, 7))

sns.barplot(x="Deposit Amount", y= "Do you own a Keurig", data= dataset)

correlation = dataset.corr()

ax = sns.heatmap(correlation,vmin=-1,vmax=1,center=0,
    cmap=sns.diverging_palette(10, 110, n=100),
    square=True)

# Data Cleaning

dataset.skew()

dataset['Deposit Amount'].fillna(100, inplace = True)
dataset.describe(include="all",datetime_is_numeric=True)

dataset.isnull().sum() 

dataset.skew()

np.unique(dataset['Ice Cream Products Consumed Per Week'])

dataset['Ice Cream Products Consumed Per Week'].fillna(0, inplace = True)

dataset.skew()
dataset['Household Income'].value_counts(dropna = False)

# Data Visualization after cleaning

dataset['Household Income'].value_counts(dropna = False).plot(kind='bar', title="Number of Trees in each Borough", figsize=(8.5, 7))

ax.set_xlabel("Household Income")
ax.set_ylabel("Number of Trees")
ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
for p in ax.patches:
    ax.annotate(str(int(p.get_height())), (p.get_x(), p.get_height()))

dataset['Household Income'].fillna("Not Specified", inplace = True)

sns.barplot(x='Household Income', y='Deposit Amount', hue='Donated To Kick Starter Before', data = dataset)

sns.boxplot(x='How many desserts do you eat a week', data = dataset)

dataset['How many desserts do you eat a week'].loc[dataset['How many desserts do you eat a week'] == 100] = 10

sns.boxplot(x='Deposit Amount', data = dataset)

