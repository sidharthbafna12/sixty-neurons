""" avg_template.py
"""

import numpy as np
from scipy.stats.mstats import mode

from clustering import NeuronClustering
from correlation import signal_correlation, noise_correlation
from scipy.signal import decimate

class AverageTemplate:
    def __init__(self, n_clusters):
        self.clustering = NeuronClustering(n_clusters, signal_correlation)

    def fit(self, tr):
        tr_data = tr.data

        # self.clustering.fit(tr_data)
        # tr_data = self.clustering.cluster_response(tr_data)

        self.templates = tr_data.mean(axis=3)
        self.n_stim = tr_data.shape[0]

    def predict(self, te):
        te_data = te.data

        # te_data = self.clustering.cluster_response(te_data)

        n_movs, n_cells, n_samples, n_trials = te_data.shape
        cell_predictions = -np.ones((n_movs, n_cells, n_trials)).astype(int)
        final_predictions = -np.ones((n_movs, n_trials)).astype(int)

        for i_s in range(n_movs):
            for i_r in range(n_trials):
                for i_n in range(n_cells):
                    r = te_data[i_s,i_n,:,i_r]
                    dists = [np.linalg.norm(r - self.templates[i_s_p,i_n,:])
                             for i_s_p in range(self.n_stim)]
                    cell_predictions[i_s,i_n,i_r] = np.argmin(dists)
                final_predictions[i_s,i_r] \
                        = mode(cell_predictions[i_s,:,i_r])[0][0].astype(int)
        return final_predictions

    def predict2(self, te):
        from fastdtw import fastdtw
        from scipy.spatial.distance import euclidean
        te_data = te.data

        # te_data = self.clustering.cluster_response(te_data)

        n_movs, n_cells, n_samples, n_trials = te_data.shape
        cell_predictions = -np.ones((n_movs, n_cells, n_trials)).astype(int)
        final_predictions = -np.ones((n_movs, n_trials)).astype(int)

        for i_s in range(n_movs):
            for i_r in range(n_trials):
                for i_n in range(n_cells):
                    r = te_data[i_s,i_n,:,i_r]
                    """
                    dists = [np.linalg.norm(r - self.templates[i_s_p,i_n,:])
                             for i_s_p in range(self.n_stim)]
                    """
                    dists = [fastdtw(r, self.templates[i_s_p,i_n,:],
                                     dist=euclidean)[0]
                             for i_s_p in range(self.n_stim)]
                    cell_predictions[i_s,i_n,i_r] = np.argmin(dists)
                final_predictions[i_s,i_r] \
                        = mode(cell_predictions[i_s,:,i_r])[0][0].astype(int)
        return final_predictions
