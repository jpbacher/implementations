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
                 
