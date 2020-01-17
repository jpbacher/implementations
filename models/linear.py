import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


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

    def _get_mse(self, y_true, y_pred):
        """MSE loss function."""
        return np.trace((y_true - y_pred).T @ (y_true - y_pred)) / y_true.shape[0]

    def plot_losses(self):
        """Plot the losses after applying gradient descent."""
        plt.figure(figsize=(6, 6))
        plt.plot(self.losses)
        sns.despine(left=True)
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.title('Loss vs Number of Iterations')


class LassoLinearRegression(LinearRegression):
    def __init__(self, lr, tolerance, iterations, lamb, pad=False):
        super().__init__(lr, tolerance, iterations, pad=False)
        self.lamb = lamb

    def fit(self, X, y):
        self.weights = np.random.randn(X.shape[1], y.shape[1])
        if self.pad:
            X = np.hstack([np.ones([X.shape[0], 1]), X])
        for i in range(self.iterations):
            y_pred = X @ self.weights
            loss = self._get_mse(y, y_pred)
            self.losses.append(loss)
            gradients = -1.0 * X.T @ (y - y_pred) + self.lamb * np.sign(self.weights)
            self.weights = self.weights - (self.lr * gradients)
            if np.abs(self.losses[-1] - self.losses[-2]) / self.losses[-1] < self.tolerance:
                break


class RidgeLinearRegression(LinearRegression):
    def __init__(self, lr, tolerance, lamb, pad=False):
        super().__init__(lr, tolerance, pad=False)
        self.lamb = lamb

    def fit(self, X, y, iterations):
        self.weights = np.random.randn(X.shape[1], y.shape[1])
        if self.pad:
            X = np.hstack([np.ones([X.shape[0], 1]), X])
        for i in range(iterations):
            y_pred = X @ self.weights
            loss = self._get_mse(y, y_pred)
            self.losses.append(loss)
            gradients = -1.0 * (y - y_pred).T @ X + self.lamb * self.weights   # 2 gets absorbed
            self.weights = self.weights - (self.lr * gradients)
            if np.abs(self.losses[-1] - self.losses[-2]) / self.losses[-1] < self.tolerance:
                break


class ElasticLinearRegression(LinearRegression):
    def __init__(self, lr, tolerance, alpha, beta, pad=False):
        super().__init__(lr, tolerance, pad=False)
        self.alpha = alpha  # seves as the lambda
        self.beta = beta

    def fit(self, X, y, iterations):
        self.weights = np.random.randn(X.shape[1], y.shape[1])
        if self.pad:
            X = np.hstack([np.ones([X.shape[0], 1]), X])
        for i in range(iterations):
            y_pred = X @ self.weights
            loss = self._get_mse(y, y_pred)
            self.losses.append(loss)
            gradients = (-1.0 * (y - y_pred).T @ X + self.alpha * self.beta * np.sign(self.weights) +
                        self.alpha * (1 - self.beta) * self.weights)
            self.weights -= self.lr * gradients
            if np.abs(self.losses[-1] - self.losses[-2]) / self.losses[-1] < self.tolerance:
                break
