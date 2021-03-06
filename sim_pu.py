import numpy as np
from sklearn.neighbors import NearestNeighbors

import torch


def elbow_curve(X, y, step=500):
    """Prepares elbow curve for choosing the k parameter."""
    expected_ones = []
    ks = []
    for k in range(500, len(y) // 4, step):
        ks.append(k)
        expected_ones.append(knn_prob(X, y, k).mean())
    return ks, expected_ones


def knn_prob(X, y, k):
    """Returns a part of y=1 among the k nearest neighbors."""
    knn = NearestNeighbors(n_neighbors=k, n_jobs=24).fit(X)
    nbh = knn.kneighbors(return_distance=False)
    return y.reshape(-1)[nbh].mean(1)


def prob_labels(y, probs):
    """Returns random labels according to probs. Leaves ones."""
    y = torch.tensor(y)
    probs = torch.tensor(probs)
    thr = np.random.sample(len(probs))
    return torch.logical_or(probs > torch.tensor(thr), y).long()
