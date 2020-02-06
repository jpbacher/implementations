from itertools import zip_longest

import numpy as np
import pandas as pd
from scipy.stats import mode


def gini(y):
    labels, frequency = np.unique(y, return_counts=True)
    t = frequency / np.sum(frequency)
    return np.sum(t * (1 - t))

def info_gain(y, mask, imp_func):
    gini = imp_func(y)
    imps = []
    unique, counts = np.unique(mask, return_counts=True)
    for i in unique:
        imps.append(imp_func(y[mask == i]))
    return gini - (np.dot(counts, imps)) / y.shape[0]

def mode(z, axis=None):
    out = mode(z, axis=axis)
    return out[0]


class Tree:
    def __init__(self,
                 X,
                 y,
                 imp_func,
                 agg_func,
                 num=50,
                 min_samples=1,
                 gain_thresh=0.1,
                 max_points=None):
        if height > 0 and X.shape[0] > min_samples:
            if max_points is None:
                max_points = X.shape[0]
            self.leaf = False
            self.gain = -1.0 * np.inf
            for feat in X.columns:
                if X[feat].dtype == 'O':
                    # get all the points in the feature
                    mask = X[feat]
                    gain = info_gain(y, mask, imp_func)
                    if gain > self.gain:
                        self.gain = gain
                        self.feat = feat
                        best_mask = mask
                else:
                    # numerical features
                    thresholds = np.linspace(X[feat].min(), X[feat].max(), num=num)
                    for i, thresh in enumerate(thresholds):
                        mask = X[feat] > thresh
                        gain = info_gain(y, mask, imp_func)
                        if gain > self.gain:
                            self.gain = gain
                            self.thresh = thresh
                            self.feat = feat
                            best_mask = mask
            self.gain_scaled = self.gain * X_shape[0]
            self.gain_scaled_norm = self.gain_scaled / max_points
            if self.gain < gain_thresh:
                self.leaf = True
                self.val = agg_func(y)
                return
            self.children = []
            self.cats = np.unique(best_mask)
            for i in self.cats:
                self.children.append(Tree(
                    X = X[best_mask == i],
                    y = y[best_mask == i],
                    height = height-1,
                    imp_func = imp_func,
                    agg_func = agg_func,
                    num=num,
                    min_samples=min_samples,
                    gain_thresh=gain_thresh,
                    max_points=max_points
                ))
        else:
            self.leaf = True
            self.val = agg_func(y)

    def predict(self, X):
        out = np.ones((X.shape[0], 1))
        if self.leaf:
            return out * self.val
        else:
            if X[self.feat].dtype != 'O':
                mask = X[self.feat] > self.thresh
            else:
                mask = X[self.feat]
            for i, c in self.enumerate(self.cats):
                out[c == mask] = self.children[i].predict(X[c == mask])
            return out
               
