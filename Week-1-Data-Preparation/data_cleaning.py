import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Dataset Inspection
titanic = pd.read_csv('train.csv')

print(titanic.head())
print(titanic.shape)
print(titanic.columns)
titanic.info()
print(titanic.describe())
print(titanic.isnull().sum())

# Data Cleaning
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
# print(titanic["Age"].median())

titanic["Embarked"] = titanic["Embarked"].fillna(titanic["Embarked"].mode()[0])

titanic = titanic.drop("Cabin", axis=1)

print(titanic.isnull().sum())

print(titanic.duplicated().sum())

print(titanic.dtypes)

titanic.to_csv("cleaned_titanic.csv",index = False)
