
# %%
from sklearn.linear_model import Ridge
import os
import math
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge, Lasso, SGDClassifier
from statsmodels.regression.linear_model import OLS
import matplotlib.pyplot as plt
# %matplotlib inline
from sklearn.model_selection import train_test_split

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

# print(x_train[:10])
# print(x_test[:10])

# print(y_train[:10])
# print(y_test[:10])

# %% [markdown]
# ## Gradient descent

# %%
class GradientDescent:
    def __init__(self):
        pass

    def fit(self, X, y, learning_rate=0.0001, iters=1000):
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.iters = iters
        self.m, self.b = self.gradient()

    def gradient(self):
        N = float(len(self.y))
        self.m, self.b = 0, 0
        b_grad, m_grad = 0, 0
        for i in range(self.iters):
            #             y_temp = np.add(np.multiply(m,self.X), b)
            # #             y_diff = np.subtract(self.y, y_temp)
            # #             print('Y_DIFF', y_diff)
            #             MSE = sum([error**2 for error in (self.y - y_temp)]) / N
            #             print('MSE', MSE)
            #             m_gradient = -(2/N) * sum(self.X * (self.y - y_temp))
            #             b_gradient = -(2/N) * sum(self.y - y_temp)
            #             m -= (self.learning_rate * m_gradient)
            #             b -= (self.learning_rate * b_gradient)
            print(len(self.X), len(self.y))
            for j in range(len(self.y)):
                print(self.y)
                x, y = self.X[j], self.y[j]
                print(x, y)
                b_grad += -(2/N) * (y - ((self.m * x) + self.b))
                m_grad += -(2/N) * x * (y - ((self.m * x) + self.b))
            self.m = - (self.learning_rate * m_grad)
            self.b = - (self.learning_rate * b_grad)
        # self.m, self.b = m, b
        return self.m, self.b  # , MSE

    def score(self, X_test, y_test):
        self.X_test = X_test
        self.y_test = y_test
        y = self.m*self.X_test + self.b
        error = 0
        for i in range(len(self.y_test)):
            x = self.X_test[i]
            y = self.y_test[i]
            error += (y - (self.m * x + self.b)) ** 2
        return error / float(len(error))

# %% [markdown]
# ## Ridge regression

# %%
# import numpy as np
# class MyRidgeRegression:
#     def __init__(self, alpha=0.1):
#         self.alpha = alpha # lambda

#     def fit(self, X, y):
#         # C = X.T.dot(X) + self.alpha*np.eye(X.shape[1])
#         self.X = X
#         self.y = y
#         # self.model = np.linalg.inv(C).dot(X.T.dot(y))
#         self.model = self._fit_model(X, y)

#     def _fit_model(self, X, y):
#         n, p = np.shape(self.X)
#         # GradientDescent(self.X, self.y, self.alpha)
#         # self.X = np.concatenate((self.X, np.sqrt(10.0**self.alpha) * np.identity(p)), axis=0)
#         # self.y = np.concatenate((self.y, np.zeros(p)), axis=0)
#         # model = SGDClassifier() #LinearRegression()
#         model = GradientDescent()
#         model.fit(X, y, learning_rate=self.alpha)
#         return model

#     def predict(self, X):
#         return X.dot(self.model)

#     def get_params(self, deep=True):
#         return {"alpha": self.alpha}

#     def set_params(self, alpha=0.1):
#         self.alpha = alpha
#         return self

#     def score(self, X, y):
#         n, p = np.shape(X)
#         X = np.concatenate(
#             (X, np.sqrt(10.0**self.alpha) * np.identity(p)), axis=0)
#         y = np.concatenate((y, np.zeros(p)), axis=0)
#         ret = self.model.score(X, y)
#         return ret


class MyRidge():

    # lmbda is for lambda

    def __init__(self, num_iters=2000, alpha=0.1, lmbda=0.1):
        self.num_iters = num_iters
        self.alpha = alpha
        self.lmbda = lmbda

    def _compute_cost(self, X, y, w, lmbda):
        """Compute the value of cost function, J.
        Here J is total Least Square Error
        """
        m = X.shape[0]
        J = (1. / (2. * m)) * \
            (np.sum((np.dot(X, w) - y) ** 2.) + lmbda * np.dot(w.T, w))

        return J

    def _gradient_descent(self, X, y, w, num_iters, alpha, lmbda):
        """Performs Graddient Descent.
        The threshold is set by num_iters, instead of some value in this implementation
        """
        m = X.shape[0]
        # Keep a history of Costs (for visualisation)
        J_all = np.zeros((num_iters, 1))

        # perform gradient descent
        for i in range(num_iters):
            #             print('GD: w: {0}'.format(w.shape))
            J_all[i] = self._compute_cost(X, y, w, lmbda)

            w = w - (alpha / m) * \
                (np.dot(X.T, (X.dot(w) - y[:, np.newaxis])) + lmbda * w)

        return w, J_all

    def fit(self, X, y):
        """Fit the model
        """
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

        self.w, self.J_all = self._gradient_descent(
            Xn, yn, w, self.num_iters, self.alpha, self.lmbda)

    def predict(self, X):
        """Predict values for given X
        """
        Xn = np.ndarray.copy(X).astype('float64')

        Xn -= self.X_mean
        Xn /= self.X_std
        Xn = np.hstack((np.ones(Xn.shape[0])[np.newaxis].T, Xn))

        return Xn.dot(self.w) + self.y_mean


my_ridge = MyRidge()
my_ridge.fit(x_train, y_train)
my_train_rsquared = my_ridge.predict(x_train)
my_test_rsquared = my_ridge.predict(x_test)

print(my_train_rsquared)


# %% [markdown]
# ## Sklearn Simple Linear Regression


# %%
linear_reg = LinearRegression()
linear_reg.fit(x_train, y_train)
train_rs2_linear = linear_reg.score(x_train, y_train)
test_rs2_linear = linear_reg.score(x_test, y_test)
print('Linear regression R^2 score: training ', train_rs2_linear)
print('Linear regression R^2 score: test ', test_rs2_linear)

# %%
lambdas = np.linspace(-3, 10, 100)
r2_train, r2_test = np.zeros(len(lambdas)), np.zeros(len(lambdas))
r2_train_impl, r2_test_impl = np.zeros(len(lambdas)), np.zeros(len(lambdas))

# %% [markdown]
# ## Comparison of lambda parameter for both implementations
# %%
# from sklearn.linear_model import Ridge
for i in range(len(lambdas)):
    model = Ridge(alpha=i, solver='sag')  # Stochastic Average Gradient
    model_impl = RidgeRegression(alpha=i)
    model.fit(x_train, y_train)
    model_impl.fit(x_train, y_train)

    r2_train[i] = model.score(x_train, y_train)
    r2_test[i] = model.score(x_test, y_test)
    r2_train_impl[i] = model_impl.score(x_train, y_train)
    r2_test_impl[i] = model_impl.score(x_test, y_test)
    print('R2_train: ', r2_train[i], 'R2_test: ', r2_test[i],
          'R2_train_impl: ', r2_train_impl[i], 'R2_train_impl: ', r2_test_impl[i])

# %%
