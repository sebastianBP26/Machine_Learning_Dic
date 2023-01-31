# -*- coding: utf-8 -*-
"""
Created on Tue Jan 24 15:51:44 2023

@author: Sebastian Barroso
"""

# Data Trasformation steps
'''When we want to apply a ML algorithms to our data, we have to make some changes in it in order
to avoid some issues (Like nas data, or stuff like that). This is guide to apply somo of the most
commum steps.'''

# Import modules
import pandas as pd

from sklearn.model_selection import train_test_split

# Preprocessing 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector as selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Data Set
heart = pd.read_csv(r'D:\ml\datasets\Heart.csv')

# Split data into X and y
X = heart.drop(['AHD'], axis = 1)
y = heart['AHD']

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.2)

# Categorical variables
categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(X_train)

# Numerical variables
numerical_columns_selector = selector(dtype_exclude=object)
numerical_columns = numerical_columns_selector(X_train)

numerical_pipeline = Pipeline([
	('imputer', SimpleImputer()),
	('scaler', StandardScaler())
])

categorical_pipeline = Pipeline([
	('encoder', OneHotEncoder())
])

full_pipeline = ColumnTransformer([
	('numeric', numerical_pipeline, numerical_columns),
	('categorical', categorical_pipeline, categorical_columns)
])

X_prepared = full_pipeline.fit_transform(X_train)
