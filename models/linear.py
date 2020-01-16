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


class MultiVariateLinearRegression:
    """Multivariate linear regression class."""
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
