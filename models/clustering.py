import numpy as np
from copy import copy
from scipy.stats import multivariate_normal


class KMeans:
    def __init__(self, k):
        self.k = k

    def fit(self, X, y=None):
        # select random initialization
        self.means = X[np.random.choice(X.shape[0], size=self.k, replace=False), :]
        old_means = None
        while (old_means != self.means).any():
            old_means = self.means
            centroid = self.predict(X)
            self.means = np.vstack([np.mean(X[centroid == k], axis=0) for k
                in range(self.k)])

    def predict(self, X):
        return np.argmin(np.sum((X - self.means.reshape([self.k, 1, X.shape[1]])) ** 2,
            axis=2), axis=0)


class GaussianMixture:
    def __init__(self, k):
        self.k = k

    def fit(self, X):
        self.means = X[np.random.choice(X.shape[0], size=self.k, replace=False), :]
        self.covs = [np.eye(X.shape[1])] * self.k
        self.covs = np.stack(self.covs)
        self.norms = [0] * self.k
        old_means = None
        old_covs = None
        i = 0
        while (old_means != self.means).any() or (old_covs != self.covs).any():
            print(f'{i}\r', end='')
            i += 1
            old_means = copy(self.means)
            old_covs = copy(self.covs)
            p_values = []
            for k in range(self.k):
                self.norms[k] = multivariate_normal(mean=self.means[k], cov=self.covs[k])
                p_values.append(self.norms[k].pdf(X))
            p_values = np.vstack(p_values)
            label = np.argmax(p_values, axis=0)
            for k in range(self.k):
                self.means[k, :] = np.mean(X[label == k], axis=0)
                self.covs[k, :, :] = np.cov(X[label == k].T)

    def predict(self, X):
        p_values = []
        for k in range(self.k):
            p_values.append(self.norms[k].pdf(X))
        p_values = np.vstack(p_values)
        label = np.argmax(p_values, axis=0)
        return label

    
