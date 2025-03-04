# %%
import pandas as pd
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import math
from machine_learning.utiles_func import *
import seaborn as sns
from sklearn.model_selection import train_test_split

# %% [markdown]
# Exercise
# Download employee retention dataset from here: https://www.kaggle.com/giripujar/hr-analytics.
# 
# Now do some exploratory data analysis to figure out which variables have direct and clear impact on employee retention (i.e. whether they leave the company or continue to work)
# Plot bar charts showing impact of employee salaries on retention
# Plot bar charts showing corelation between department and employee retention
# Now build logistic regression model using variables that were narrowed down in step 1
# Measure the accuracy of the model

# %%
df = pd.read_csv('data/HR_comma_sep.csv')
df.head()

# %%
df['left'].value_counts()

# %%
df.select_dtypes(['float','int']).groupby('left').mean()

# %%
# Above bar chart shows employees with high salaries are likely to not leave the company
pd.crosstab(df.salary,df.left).plot(kind='bar')

# %%
pd.crosstab(df.Department,df.left).plot(kind='bar')

# %%
subdf = df[['satisfaction_level','average_montly_hours','promotion_last_5years','salary','left']]
subdf.head()

# %%
subdf = encode_categorical_and_split(subdf, random_state=1234)


# %%
subdf['Set'].unique()

# %%
X_train = subdf[subdf['Set']=="Train"].drop(['left','Set'],axis=1)
X_train.head()

# %%
X_test = subdf[subdf['Set']=="Test"].drop(['left','Set'],axis=1)
X_test.head()

# %%
y_train = subdf[subdf['Set']=="Train"]['left']
y_test = subdf[subdf['Set']=="Test"]['left']

# %%
print(
    X_train.shape,
    X_test.shape,
    y_train.shape,
    y_test.shape,
)

# %%
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

# %%
model.fit(X_train, y_train)

# %%
model.predict(X_test)

# %%
model.score(X_test,y_test)


