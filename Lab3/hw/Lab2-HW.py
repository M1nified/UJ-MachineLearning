# %%

import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.linear_model import Lasso, LinearRegression, Ridge, SGDClassifier
from sklearn.metrics import precision_score, r2_score
from sklearn.model_selection import cross_validate, train_test_split
from statsmodels.regression.linear_model import OLS

# http://archive.ics.uci.edu/ml/datasets/Wine+Quality
np.random.seed(42)

# %%
data = pd.read_csv(
    'http://cs.if.uj.edu.pl/piotrek/ML2019/datasets/dataset_2.txt')
data.info()
# data['make']


# %%

y = data['price']
X = data.drop('price', axis=1)
X = X[['horsepower', 'wheel-base', 'width', 'height', 'curb-weight',
       'engine-size', 'compression-ratio', 'peak-rpm', 'city-mpg', 'highway-mpg']]

x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=2000)

x_train = x_train.values
x_test = x_test.values
y_train = y_train.values
y_test = y_test.values

# %% [markdown]
# ## Custom Ridge Class

# %%


class MyRidge(BaseEstimator):

    # lmbda is for lambda

    def __init__(self, iter_count=2000, alpha=0.1, lmbda=0.1, verbose=False):
        self.iter_count = iter_count
        self.alpha = alpha
        self.lmbda = lmbda
        self.verbose = verbose

    def _compute_cost(self, X, y, w, lmbda):
        """Compute the value of cost function, J.
        Here J is total Least Square Error
        """
        m = X.shape[0]
        J = (1. / (2. * m)) * \
            (np.sum((np.dot(X, w) - y) ** 2.) + lmbda * np.dot(w.T, w))

        return J

    def _gradient_descent(self, X, y, w, iter_count, alpha, lmbda):
        """Performs Graddient Descent.
        The threshold is set by iter_count, instead of some value in this implementation
        """
        m = X.shape[0]
        # Keep a history of Costs (for visualisation)
        b = np.zeros((iter_count, 1))

        # perform gradient descent
        for i in range(iter_count):
            #             print('GD: w: {0}'.format(w.shape))
            b[i] = self._compute_cost(X, y, w, lmbda)

            w = w - (alpha / m) * \
                (np.dot(X.T, (X.dot(w) - y[:, np.newaxis])) + lmbda * w)

        return w, b

    def fit(self, X, y):
        Xn = np.ndarray.copy(X).astype('float64')
        yn = np.ndarray.copy(y).astype('float64')

        # initialise w params for linear model, from w0 to w_num_features
        w = np.zeros((Xn.shape[1] + 1, 1))

        # normalise the X
        self.X_mean = np.mean(Xn, axis=0)
        self.X_std = np.std(Xn, axis=0)
        Xn -= self.X_mean
        self.X_std[self.X_std == 0] = 1
        Xn /= self.X_std

        self.y_mean = yn.mean(axis=0)
        yn -= self.y_mean

        # add ones for intercept term
        Xn = np.hstack((np.ones(Xn.shape[0])[np.newaxis].T, Xn))

        self.w, self.b = self._gradient_descent(
            Xn, yn, w, self.iter_count, self.alpha, self.lmbda)

    def predict(self, X):
        Xn = np.ndarray.copy(X).astype('float64')

        Xn -= self.X_mean
        Xn /= self.X_std
        Xn = np.hstack((np.ones(Xn.shape[0])[np.newaxis].T, Xn))

        predicted = Xn.dot(self.w) + self.y_mean
        predicted = [elem for sublist in predicted for elem in sublist]

        return predicted

    def _sums(self, X, y, y_predicted):
        ss_tot = 0
        ss_reg = 0
        ss_res = 0
        y_avg = sum(y)/len(y)
        for i in range(len(y)):
            y_i = y[i]
            f_i = y_predicted[i]
            ss_tot += (y_i - y_avg)**2
            ss_reg += (f_i - y_avg)**2
            ss_res += (y_i - f_i)**2
        return ss_tot, ss_reg, ss_res

    def r_square(self, y, y_predicted):
        ss_tot, ss_reg, ss_res = self._sums(_, y, y_predicted)
        r = 1 - (ss_res/ss_tot)
        return r

    def score(self, y, y_predicted):
        return precision_score(y, y_predicted)
        # err = 0
        # for i in range(len(y)):
        #     x_i = X[i]
        #     y_i = y[i]
        #     print(x_i, y_i)
        #     err += (y_i-(self.w * x_i + self.b))**2
        # return err


# %%
my_ridge = MyRidge()
my_ridge.fit(x_train, y_train)
my_ridge_predictions = my_ridge.predict(x_test)
my_ridge_predictions

my_ridge_r_square = my_ridge.r_square(y_test, my_ridge_predictions)
print(my_ridge_r_square)
my_ridge_r_square_sklearn = r2_score(y_test, my_ridge_predictions)
print(my_ridge_r_square_sklearn)

# my_ridge_cross_validation = cross_validate(MyRidge(), x_test, y_test)
# print(my_ridge_cross_validation)

# %% [markdown]
# ## Sklearn Simple Linear Regression

# %%
sklearn_linear_regression = LinearRegression()
sklearn_linear_regression.fit(x_train, y_train)
sklearn_linear_regression_predictions = sklearn_linear_regression.predict(
    x_test)
sklearn_linear_regression_predictions

# %% [markdown]
# ## Sklearn Ridge Regression using Stochastic Average Gradient

# %%
sklearn_ridge = Ridge(solver='sag')
sklearn_ridge.fit(x_train, y_train)
sklearn_ridge_predictions = sklearn_ridge.predict(x_test)
sklearn_ridge_predictions

# %% [markdown]
# ## All in one
# %%
plt.scatter(y_test, my_ridge_predictions,
            color='g', alpha=0.8, label='MyRidge')
plt.scatter(y_test, sklearn_linear_regression_predictions,
            color='r', alpha=0.4, label='sklearn LinearRegression')
plt.scatter(y_test, sklearn_ridge_predictions, color='b',
            alpha=0.4, label='sklearn Ridge(solver=\'sag\')')
plt.plot([0, 40000], [0, 40000], 'y-', lw=2)

plt.legend()
plt.show()
# %% [markdown]
# ## MyRidge vs Sklearn's LinearRegression
# %%
plt.scatter(y_test, my_ridge_predictions,
            color='g', alpha=0.8, label='MyRidge')
plt.scatter(y_test, sklearn_linear_regression_predictions,
            color='r', alpha=0.4, label='sklearn LinearRegression')
plt.plot([0, 40000], [0, 40000], 'y-', lw=2)

plt.legend()
plt.show()
# %% [markdown]
# ## MyRidge vs Sklearn's Ridge using Stochastic Average Gradient
# %%
plt.scatter(y_test, my_ridge_predictions,
            color='g', alpha=0.8, label='MyRidge')
plt.scatter(y_test, sklearn_ridge_predictions, color='b',
            alpha=0.4, label='sklearn Ridge(solver=\'sag\')')
plt.plot([0, 40000], [0, 40000], 'y-', lw=2)

plt.legend()
plt.show()


# %%
