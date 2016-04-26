#!/usr/bin/env python
""" single_trial_classify.py
    Classify a single trial V1 response as one of the possible stimulus types as
    described in the training data.

    There is not that much data that one can try particularly complicated
    things, so a nearest-neighbour/template-matching approach is probably the
    most sensible option.
"""

################################################################################
# Basics
import numpy as np
from matplotlib import pyplot as plt
import os

from src.response import Response

# For classification and for confusion matrix
from src.nn_template import ClusterTemplateNN
from sklearn.metrics import confusion_matrix

from src.correlation import signal_correlation, noise_correlation
import scipy.cluster.hierarchy as hac
from scipy.spatial.distance import squareform

train_frac = 0.5
n_clusters = 3
exp_type = 'natural'
if exp_type == 'grating':
    from src.params.grating.datafile_params import *
    from src.params.grating.stimulus_params import *
    data_locs = [os.path.join(DATA_DIR, '%s_dir.npy' % c) for c in MICE_NAMES]
    data = map(lambda (n, loc) : Response(n, loc), zip(MICE_NAMES, data_locs))
elif exp_type == 'natural':
    from src.params.naturalmovies.datafile_params import *
    from src.params.naturalmovies.stimulus_params import *
    data_locs = [os.path.join(DATA_DIR, '%d.npy' % i) for i in range(11)]
    data = [Response(str(i), data_locs[i]) for i in range(11)]

def grating_cm_goodness(m):
    L = m.shape[0]

for index, m in enumerate(data):
    print 'Mouse %s' % m.name
    
    S, N, T, N_TR = m.data.shape
    n_train = int(train_frac * N_TR)

    m.stc_train = m.data[:,:,:,:n_train]
    m.stc_test = m.data[:,:,:,n_train:]
    
    m.stc_model = ClusterTemplateNN(K=10)
    m.stc_model.fit(m.stc_train)
    m.stc_true_labels = np.repeat(np.arange(S), N_TR - n_train)\
                          .reshape((S, N_TR - n_train))
    m.stc_pred_labels = m.stc_model.predict(m.stc_test)
    m.stc_conf_mats = [confusion_matrix(m.stc_true_labels.flatten(),
                                        m.stc_pred_labels[:,:,i_n].flatten())
                       for i_n in range(N)]
    m.avg_stc_conf_mat_goodness = np.mean([float(np.diag(c).sum()) / c.sum()
                                           for c in m.stc_conf_mats])

    # Also try the same analysis but with clustered responses.
    m.signal_corr = signal_correlation(m.data)
    dists_sc = squareform(0.5 * (1.0 - m.signal_corr))
    m.linkage_sc = hac.linkage(dists_sc, method='complete')
    m.cl_idxs_sc = hac.fcluster(m.linkage_sc, n_clusters, criterion='maxclust')
    m.sc_clust_rsp = np.zeros((S, n_clusters, T, N_TR))
    for i in range(n_clusters):
        i_cl = i+1
        idxs = np.arange(N)
        idxs = idxs[m.cl_idxs_sc == i_cl]
        cluster_mean = m.data[:,idxs,:,:].mean(axis=1)
        m.sc_clust_rsp[:,i,:,:] = cluster_mean
    m.sc_clust_train = m.sc_clust_rsp[:,:,:,:n_train]
    m.sc_clust_test = m.sc_clust_rsp[:,:,:,n_train:]
    m.stc_model_sc_clust = ClusterTemplateNN(K=3)
    m.stc_model_sc_clust.fit(m.sc_clust_train)
    m.stc_pred_labels_sc_clust = m.stc_model_sc_clust.predict(m.sc_clust_test)
    m.stc_conf_mats_sc_clust = \
            [confusion_matrix(m.stc_true_labels.flatten(),
                              m.stc_pred_labels_sc_clust[:,:,i_c].flatten())
             for i_c in range(n_clusters)]
    m.avg_stc_conf_mat_sc_clust_goodness = \
                                    np.mean([float(np.diag(c).sum())/c.sum() 
                                             for c in m.stc_conf_mats_sc_clust])

    m.noise_corr = noise_correlation(m.data)
    dists_nc = squareform(0.5 * (1.0 - m.signal_corr))
    m.linkage_nc = hac.linkage(dists_nc, method='complete')
    m.cl_idxs_nc = hac.fcluster(m.linkage_nc, n_clusters, criterion='maxclust')
    m.nc_clust_rsp = np.zeros((S, n_clusters, T, N_TR))
    for i in range(n_clusters):
        i_cl = i+1
        idxs = np.arange(N)
        idxs = idxs[m.cl_idxs_nc == i_cl]
        cluster_mean = m.data[:,idxs,:,:].mean(axis=1)
        m.nc_clust_rsp[:,i,:,:] = cluster_mean
    m.nc_clust_train = m.nc_clust_rsp[:,:,:,:n_train]
    m.nc_clust_test = m.nc_clust_rsp[:,:,:,n_train:]
    m.stc_model_nc_clust = ClusterTemplateNN(K=3)
    m.stc_model_nc_clust.fit(m.nc_clust_train)
    m.stc_pred_labels_nc_clust = m.stc_model_nc_clust.predict(m.nc_clust_test)
    m.stc_conf_mats_nc_clust = \
            [confusion_matrix(m.stc_true_labels.flatten(),
                              m.stc_pred_labels_nc_clust[:,:,i_c].flatten())
             for i_c in range(n_clusters)]
    m.avg_stc_conf_mat_nc_clust_goodness = \
                                    np.mean([float(np.diag(c).sum())/c.sum() 
                                             for c in m.stc_conf_mats_nc_clust])
