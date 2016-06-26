""" dtw_nn.py
    A nearest-neighbour classifier with the DTW algorithm used to compute the
    distance between sequences.

    Computes distances only for a specified number of randomly sampled
    sequences from each class.
"""

import numpy as np
from scipy.stats.mstats import mode

import random

from fastdtw import fastdtw
from scipy.spatial.distance import euclidean, cityblock

class DTWClassifier:
    def __init__(self, K):
        self.K = K
    
    def _d(self, r1, r2):
        return np.sum(np.abs(r1 - r2))
    
    """
        Not being used now.
    def _dtw(self, s1, s2):
        N = s1.shape[0]
        M = s2.shape[0]
        dist = np.zeros((N+1,M+1))

        for i in range(1, N+1):
            dist[i,0] = np.inf
        for i in range(1, M+1):
            dist[0,i] = np.inf
        dist[0,0] = 0

        for i in range(1, N+1):
            for j in range(1, M+1):
                cost = self._d(s1[i-1,:], s2[j-1,:])
                dist[i,j] = cost + min(dist[i-1,j],dist[i,j-1],dist[i-1,j-1])
        return dist[N,M]
    """

    def fit(self, tr_features):
        # tr_features is expected to be organised already; tr_features[i] should
        # contain the training instances from class i.
        self.tr_seqs = tr_features

    def nearest(self, rsp):
        test_seq = rsp
        all_dists = [[fastdtw(test_seq, r, dist=euclidean)[0]
                      for r in random.sample(class_seqs, self.K)]
                     for class_seqs in self.tr_seqs]
        min_dists = [min(dists) for dists in all_dists]
        return np.argmin(min_dists)

    def predict(self, te_features):
        # te_features is supposed to be organised already; it is expected to be
        # a list of lists: te_features[i] contains the test samples whose true
        # label is that of the ith class.
        final_predictions = []

        for rsp_set in te_features:
            predictions = []
            for rsp in rsp_set:
                N = rsp.shape[1]
                predictions.append(self.nearest(rsp))
                print '\t\t%d' % predictions[-1]
            final_predictions.append(predictions)
        return final_predictions
