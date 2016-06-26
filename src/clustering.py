""" clustering.py
    Computes clustering using a given correlation function as a distance metric.
    Uses the distance metric for clustering with the hierarchical agglomerative
    clustering.

    The clustering can be used for reducing dimensionality (using cluster mean
    as representative), divisive normalisation, etc.
"""

import numpy as np

import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import squareform

class NeuronClustering:
    def __init__(self, n_clusters, corr_func):
        self.n_clusters = n_clusters
        self.corr = corr_func
    
    def fit(self, data):
        C = self.corr(data)
        dists = squareform(0.5 * (1.0 - C))
        self.linkage = hac.linkage(dists, method='complete')
        self.cl_idxs = hac.fcluster(self.linkage, self.n_clusters,
                                    criterion='maxclust')
        self.n_stim, self.n_cells, self.n_samples, self.n_trials = data.shape

    def divnorm(self, data):
        # Normalise the responses by the sum of the responses in the clusters.
        n_stim, n_cells, n_samples, n_trials = data.shape
        rsp = np.copy(data)

        for i_c in range(1, self.n_clusters + 1):
            idxs = [i for i in range(n_cells) if self.cl_idxs[i] == i_c]

            for tr in range(n_trials):
                for t in range(n_samples):
                    for i_s in range(n_stim):
                        if rsp[i_s,idxs,t,tr].sum() == 0.0:
                            continue
                        rsp[i_s,idxs,t,tr] /= rsp[i_s,idxs,t,tr].sum()
        return rsp
    
    def reduction_fn(self, cl_rsp):
        return np.mean(cl_rsp, axis=1)

    def cluster_response(self, data):
        n_stim, n_cells, n_samples, n_trials = data.shape
        cl_rsps = np.zeros((n_stim, self.n_clusters, n_samples, n_trials))
        for i_c in range(1, self.n_clusters + 1):
            idxs = np.arange(self.n_cells)
            idxs = idxs[self.cl_idxs == i_c]
            cl_rsp = data[:,idxs,:,:]
            cl_rsps[:,i_c-1,:,:] = self.reduction_fn(cl_rsp)
        return cl_rsps
