# data analysis and wrangling
import keras
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

train_df = pd.read_csv("./train.csv")
test_df = pd.read_csv("./test.csv")
combine = [train_df, test_df]

print(train_df.columns.values)

train_df.head()
train_df.tail()
train_df.info()

print(train_df)

print(train_df.columns)

print(train_df.columns.values)

train_df.head(8)

train_df.describe()

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False)

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=False).mean()

train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=False).mean()

train_df[['Sex', 'Survived']].groupby(['Sex'], as_index=True).mean()

train_df[['Pclass', 'Survived']].groupby(['Pclass'], as_index=True).mean()

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=20)

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=10)

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=40)

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=200)

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=5)

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=0)

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=1)

g = sns.FacetGrid(train_df, col='Survived')
g.map(plt.hist, 'Age', bins=2)

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend()

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.1, bins=20)
grid.add_legend()

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=1, bins=20)
grid.add_legend()

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)

grid = sns.FacetGrid(train_df, col='Survived', row='Pclass', height=2.2, aspect=1.6)
grid.map(plt.hist, 'Age', alpha=.5, bins=20)
grid.add_legend();