import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
import matplotlib.pyplot as plt
import seaborn as sns
np.random.seed(12)


def binarycrossentropy(y, p_hat):
    return -1.0 * np.sum(y * np.log(p_hat) + (1 - y) * np.log(1 - p_hat))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def mse(y_true, y_pred):
    """MSE loss function."""
    return np.trace((y_true - y_pred).T @ (y_true - y_pred)) / y_true.shape[0]

class UnivariateLinearRegression:
    """Linear regression class with 1 variable."""
    def __init__(self):
        self.beta_0 = 0
        self.beta_1 = 0

    def fit(self, x, y):
        x_bar = np.mean(x)
        y_bar = np.mean(y)
        xx_bar = np.mean(x**2)
        xy_bar = np.mean(x*y)

        self.beta_1 = x_bar * y_bar - xy_bar / (x_bar**2 - xx_bar)
        self.beta_0 = y_bar - x_bar * beta_1

    def predict(self, x):
        y_pred = self.beta_0 + self.beta_1 * x
        return y_pred


class LinearRegressionNormal:
    """Multivariate linear regression using the Normal Equation class."""
    def __init__(self, pad=False):
        self._pad = pad
        self.weights = 0

    def fit(self, X, y):
        assert X.shape[1] > 1, "Use Univariate Linear Regression class"
        if self._pad:
            X = self._get_ones(X)
        self.weights = np.linalg.inv(X.transpose().dot(X)).dot(X.transpose()).dot(y)

    def predict(self, X):
        if self._pad:
            X = self._get_ones(X)
        predictions = X.dot(self.weights)
        return predictions

    def _get_ones(self, X):
        ones = np.ones(X.shape[0]).reshape(-1, 1)  # get ones for each instance
        X = np.concatenate((ones, X), axis=1)
        return X


class LinearRegression():
    def __init__(self, n_iter=10000, tol=1e-4, lr=1e-4, fit_intercept=True):
        self.n_iter = n_iter
        self.tol = tolerance
        self.lr = lr
        self.fit_intercept = fit_intercept

    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.hstack([intercept, X])

    def fit(self, X, y):
        """
        Apply gradient descent based on number of iterations or until loss
        stops improving more than specified tolerance level.
        """
        self.losses = [np.inf]
        self.weights_ = np.random.randn(X.shape[1], y.shape[1])
        if self.fit_intercept:
            X = self._add_intercept(X)
        for i in range(self.n_iter):
            y_pred = X @ self.weights_
            loss = mse(y, y_pred)
            self.losses.append(loss)
            gradients = -1.0 * X.T @ (y - y_pred)
            self.weights_ = self.weights_ - (self.lr * gradients)
            if np.abs(self.losses[-1] - self.losses[-2]) / self.losses[-1] < self.tol:
                break

    def partial_fit(self, X, y):
        """
        This method gets X and y arrays and applies gradient descent. It does not
        initialize weights every time so we can feed the algorithm different data
        points; only runs one iteration.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input sample.
        y : array, shape (n_samples,)
            The target variable, a real number.
        Returns
        -------
        loss : loss value of one iteration.
        """

        X, y = check_X_y(X, y)
        self.losses = [np.inf]
        if self.intercept:
            X = self._add_intercept(X)
        if self.weights_ is None:
            self.weights_ = np.random.randn(X.shape[1], y.shape[1])
        y_pred = X @ self.weights_
        loss = mse(y, y_pred)
        self.losses.append(loss)
        gradients = -1.0 * X.T @ (y - y_pred)
        self.weights_ -= (self.lr * gradients)
        loss = mse(y, y_pred)
        return loss

    def predict(self, X):
        if self.fit_intercept:
            X = self._add_intercept(X)
        y_pred = X @ self.weights
        return y_pred

    def plot_losses(self):
        """Plot the losses after applying gradient descent."""
        plt.figure(figsize=(6, 6))
        plt.plot(self.losses)
        sns.despine(left=True)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss vs Number of Iterations')


class LogisticRegression():
    def __init__(self, n_iter=10000, tol=1e-4, lr=1e-4, alpha=0, beta=0,
                fit_intercept):
        self.n_iter = n_iter
        self.tolerance = tol
        self.lr = lr
        self.alpha = alpha
        self.beta = beta
        self.fit_intercept = fit_intercept

    def _add_intercept(self, X):
        intercept = np.ones((X.shape[0], 1))
        return np.hstack([intercept, X])

    def partial_fit(self, X, y):
        """
        This method gets X and y arrays and applies gradient descent. It does not
        initialize weights every time so we can feed the algorithm different data
        points; only runs one iteration.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input sample.
        y : array, shape (n_samples,)
            The target values, an array of ones and zeros.
        Returns
        -------
        loss : loss value of one iteration.
        """
        self.weights_ = None
        X, y = check_X_y(X, y)
        if self.fit_intercept:
            X = self._add_intercept(X)
        if self.weights_ is None:
            self.weights_ = np.random.randn(X.shape[1], y.shape[1])
        p_pred = self.predict_proba(X)
        grad = -1.0 * X.T @ (y - p_pred) + \
                self.alpha * (
                    self.beta * np.sign(self.weights_) +
                    (1 - self.beta) * self.weights_
                )
        self.weights_ -= self.lr * grad
        loss = binarycrossentropy(y, p_pred)
        return loss 

    def fit(self, X, y):
        """
        This method gets X and y arrays and applies gradient descent.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The training input samples.
        y : array, shape (n_samples,)
            The target values, an array of ones and zeros.
        Returns
        -------
        self : object
        """
        X, y = check_X_y(X, y)
        loss = [np.inf]
        if self.fit_intercept:
            X = self._add_intercept(X)
        self.weights_ = np.random.randn(X.shape[1], y.shape[1])
        for i in range(self.n_iter):
            p_hat = self.predict_proba(X)
            loss.append(binarycrossentropy(y, p_hat))
            grad = -1.0 * X.T @ (y - p_hat) + \
                self.alpha * (
                    self.beta * np.sign(self.weights_) +
                    (1 - self.beta) * self.weights_
                )
            self.weights_ -= self.lr * grad
            if abs(loss[-1] - loss[-2]) < self.tolerance:
                break
        return self

    def predict_proba(self, X):
        """
        Get predicted probabilities.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted probabilities for each sample.
        """
        # check is fit has been called
        # check_is_fitted(self, 'is_fitted_')
        # input validation
        X = check_array(X, accept_sparse=True)
        if self.fit_intercept:
            X = self._add_intercept(X)
        y_hat = X @ self.weights_
        return sigmoid(y_hat)

    def predict(self, X, thresh=0.5):
        """
        Get hard-coded predictions from predicted probabilities determined
        by threshold.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The input samples.
        thresh : float, between 0 and 1
        Returns
        -------
        y : ndarray, shape (n_samples,)
            The predicted class for each sample.
        """
        return self.predict_proba(X) >= thresh
