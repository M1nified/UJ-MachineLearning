
# %%
import os
import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDClassifier
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split

# %%
data = pd.read_csv('http://cs.if.uj.edu.pl/piotrek/ML2019/datasets/dataset_2.txt')
data.info()
# print(data[:2])

# Split predictors and response
# x = data[:, :-1]
# y = data[:, -1]

# df = pd.DataFrame(data)
# df.head()
data['make']

# %%



