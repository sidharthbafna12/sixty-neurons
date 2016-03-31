#!/usr/bin/python

# Experiment/plotting/other parameters
from params.grating.datafile_params import *
from params.grating.stimulus_params import *

# Basics
import numpy as np
from scipy.stats import mode

# Clustering
import scipy.cluster.vq as scvq

# Data format
from grating_response import GratingResponse

class GratingClusterTemplateNN:
    K = 10

    def __init__(self):
        pass

    def fit(self, tr):
        T = GRATING_DURATION * CA_SAMPLING_RATE
        N = tr.shape[3]
        N_tr = tr.shape[1]
        # rsps = np.mean(tr, axis=GratingResponse.TrialAxis) # (16, 40, N)
        rsps = tr.swapaxes(1,3).swapaxes(1,2)\
                 .reshape((len(ORIENTATIONS),T,N_tr*N))
        self.centroids, self.cluster_labels = [], []
        for i in range(len(ORIENTATIONS)):
            rsps_i = rsps[i,:,:].swapaxes(0,1)
            ce, la = scvq.kmeans2(rsps_i, GratingClusterTemplateNN.K,
                                  minit='points')
            la = [int(mode(la[n:n+N_tr])[0][0])
                  for n in range(0,N_tr*N,N_tr)]
            self.centroids.append(ce)
            self.cluster_labels.append(la)
        self.templates = np.array(self.centroids)

    def predict(self, te):
        N = te.shape[GratingResponse.CellsAxis]
        n_trials = te.shape[GratingResponse.TrialAxis]
        n_dirs = te.shape[GratingResponse.DirAxis]
        
        labels = np.zeros((n_dirs, n_trials, N)).astype(int)
        for i in range(N):
            for i_trial in range(n_trials):
                for i_dir in range(n_dirs):
                    r = te[i_dir, i_trial, :, i]
                    cl_idxs = [self.cluster_labels[j_dir][i]
                               for j_dir in range(n_dirs)]
                    dists = [np.linalg.norm(r - self.templates[j,cl_idxs[j],:])
                             for j in range(n_dirs)]
                    labels[i_dir, i_trial, i] = np.argmin(dists)
        return labels

class NatMoviesClusterTemplateNN:
    K = 3

    def __init__(self):
        pass

    def fit(self, tr):
        n_movies = len(tr)
        self.centroids, self.cluster_labels = [], []
        for i in range(n_movies):
            avg = np.mean(tr[i], axis=2) # 49 x 200
            ce, la = scvq.kmeans2(avg, NatMoviesClusterTemplateNN.K,
                                  minit='points')
            self.centroids.append(ce)
            self.cluster_labels.append(la)
        self.templates = np.array(self.centroids)

    def predict(self, te):
        n_movies = len(te)
        n_neurons = te[0].shape[0]
        all_labels = []
        for i_n in range(n_neurons):
            labels = []
            for i_m in range(n_movies):
                n_trials = te[i_m].shape[2]
                for i_trial in range(n_trials):
                    r = te[i_m][i_n, :, i_trial]
                    cl_idxs = [self.cluster_labels[j_m][i_n]
                               for j_m in range(n_movies)]
                    dists = [np.linalg.norm(r - self.templates[j,cl_idxs[j],:])
                             for j in range(n_movies)]
                    labels.append(np.argmin(dists))
            all_labels.append(labels)
        return all_labels
