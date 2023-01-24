# -*- coding: utf-8 -*-
"""
Created on Mon Jan 23 21:35:15 2023

@author: Sebastian Barroso
"""

# 8.3.3 (ISL) Bagging and Random Forest Lab.

#==================================================================================================
# modules
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor 
from sklearn.tree import DecisionTreeRegressor

#==================================================================================================

# Load data
boston = load_boston()
t = boston['data']
y = boston['target']

# split data in train and test
x_train, x_test, y_train, y_test = train_test_split(t, y, test_size=0.3)

#==================================================================================================
# Model 1:
# RandomForest: Using 500 Trees all the features.
rf = RandomForestRegressor(n_estimators = 500)
rf.fit(x_train, y_train)
y_hat = rf.predict(x_test)
np.mean((y_hat - y_test)**2) # Error estimate

# Model 2:
# Decision Tree
dt = DecisionTreeRegressor(max_depth = 15)
dt.fit(x_train, y_train)
y_hat = dt.predict(x_test)
np.mean((y_hat - y_test)**2) # Error estimate
   
# Model 3: 
# RandomForest: Using 1000 trees 6 features per tree.
rf = RandomForestRegressor(n_estimators = 1000, max_features = 6)
rf.fit(x_train, y_train)
y_hat = rf.predict(x_test)
np.mean((y_hat - y_test)**2)

# We note that using a RF we can get a lowe error than using a DT.

# Feature importances:
importance = pd.DataFrame({'variable': boston['feature_names'],
                           'importance': rf.feature_importances_})

# We note that there are some variables that are more important than others
figure, ax = plt.subplots(nrows = 1, ncols = 1, dpi = 200, figsize = (10,8))
importance.sort_values(['importance']).plot(x = 'variable', y = 'importance', kind = 'bar', ax = ax)