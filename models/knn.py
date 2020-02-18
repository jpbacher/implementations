from collections import Counter
import numpy as np


class Knn:
    def __init__(self, k):
        self.k = k

    def _get_distance(self, point1, point2):
        point1 = np.array(point1)
        point2 = np.array(point2)
        return np.linalg.norm(point1 - point2)

    def fit(self, X, labels, test_instance):
        distances = []
        for idx in range(len(X)):
            distance = self._get_distance(X[idx], test_instance)
            distances.append((X[idx], distance, labels[idx]))
        distances.sort(key=lambda x: x[1])
        self.nearest_neighbors = distances[:self.k]
        return self.nearest_neighbors

    def predict(self):
        class_counter = Counter()
        for neighbor in self.nearest_neighbors:
            class_counter[neighbor[2]] += 1
        return class_counter.most_common(1)[0][0]

    
