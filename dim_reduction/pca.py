import numpy as np


def PCA:
    def __init__(self, p=None, num=None):
        self.num = num
        self.p = p

    def fit(self, X):
        self.eig_val, self.eig_vec np.linalg.eig(X.T @ X)
        self.eig_val = np.abs(self.eig_val)
        ind = np.argsort(self.eig_val)[::-1]
        self.eig_val = self.eig_val[ind]
        self.eig_vec = self.eig_vec[ind]
        if self.p:
            self.num = np.sum(np.cumsum(pca.eig_val / np.sum(
                pca.eig_val)) < self.p)

    def transform(self, X):
        return X @ self.eig_vec[:, :self.num]

    def fit_transform(self, X):
        return self.fix(X).transform(X)

    def filter(self, X):
        return X @ self.eig_vec[:, :self.num] @ self.eig_vec[:, :self.num].T
