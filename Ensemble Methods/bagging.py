# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 16:34:21 2023

@author: Sebastian Barroso
"""

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, cross_validate
from sklearn.compose import make_column_selector as selector
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier

# Baggin: Boostrap Aggregation Example
# ==================================================================================================
df = pd.read_csv(r'D:\ml\datasets\Heart.csv')
df.drop(['Unnamed: 0'], axis = 1, inplace = True)
df.isna().sum()
# delete the null data
df = df.dropna()

X = df.drop(['AHD'], axis = 1)
y = df['AHD']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

categorical_columns_selector = selector(dtype_include = object)
categorical_columns = categorical_columns_selector(X_train)

numerical_columns_selector = selector(dtype_exclude= object)
numerical_columns = numerical_columns_selector(X_train)

# this pipeline will help us to transform the data in the dataset
preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(), categorical_columns)
    ])

rf = RandomForestClassifier(n_estimators = 100, 
                               max_depth = 20,
                               max_features = 'sqrt',
                               n_jobs = 2)

# In order to evaluate, we add the preprocessor and the random forest classifier in a pipeline
model = make_pipeline(preprocessor, rf)
results = cross_validate(model, X_test, y_test, 
                         scoring = 'accuracy', 
                         cv = 10, 
                         return_train_score = True)

print(np.round(results['train_score'].mean(), 2))
print(np.round(results['test_score'].mean(), 2))