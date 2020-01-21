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
    def __init__(self, lr, tolerance, iterations, pad=False):
        self.lr = lr
        self.tolerance = tolerance
        self.iterations = iterations
        self.pad = pad
        self.losses = [np.inf]
        self.weights = 0

    def fit(self, X, y):
        """
        Apply gradient descent based on number of iterations or until loss
        stops improving more than specified tolerance level.
        """
        self.weights = np.random.randn(X.shape[1], y.shape[1])
        if self.pad:
            X = np.hstack([np.ones([X.shape[0], 1]), X])
        for i in range(self.iterations):
            y_pred = X @ self.weights
            loss = self._get_mse(y, y_pred)
            self.losses.append(loss)
            gradients = -1.0 * X.T @ (y - y_pred)
            self.weights = self.weights - (self.lr * gradients)
            if np.abs(self.losses[-1] - self.losses[-2]) / self.losses[-1] < self.tolerance:
                break

    def predict(self, X):
        if self.pad:
            X = np.hstack([np.ones([X.shape[0], 1]), X])
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


class LogisticRegression(BaseEstimator, ClassifierMixin):
    def __init__(self, n_iter=10000, tol=1e-4, lr=1e-4, alpha=0, beta=0,
                fit_intercept):
        self.n_iter = n_iter
        self.tolerance = tol
        self.lr = lr
        self.weights_ = 0
        self.alpha = alpha
        self.beta = beta
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        """
        This method gets X and y arrays and applies gradient descent.

        :param X: nd array
        :param y: nd array
        :param pad: boolean arguement to add y-intercept

        :return: self.beta
        """
        X, y = check_X_y(X, y)
        loss = [np.inf]
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        self.weights_ = np.random.randn(X.shape[1], y.shape[1])
        for i in range(self.n_iter):
            p_hat = self.predict(X)
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

    def predict(self, X):

        check_is_fitted(self)
        X = check_array(X)
        if self.fit_intercept:
            X = np.hstack([np.ones((X.shape[0], 1)), X])
        y_hat = X @ self.weights_
        p_hat = sigmoid(y_hat)
        return p_hat
