#!/usr/bin/python
""" clustering.py
    Cluster the average responses of mouse neurons to different sinusoidal
    grating orientations to see which sets of neurons seem to fire together.

    This proceeds using the method of hierarchical agglomerative clustering
    which reorders the correlation matrix of neuron responses to collect similar
    neurons together. This can be thresholded to produce a flat set of clusters.
"""

# Experiment/plotting/other parameters
from src.params.grating.datafile_params import *
from src.params.grating.stimulus_params import *

# Basics
import numpy as np
from matplotlib import pyplot as plt
import os

# clustering
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as hac

# Reading in the data
from src.response import Response

data_locs= [os.path.join(DATA_DIR, '%s_ori.npy' % c) for c in MICE_NAMES]
data = map(lambda (n, loc): Response(n, loc), zip(MICE_NAMES, data_locs))

for index, m in enumerate(data):
    name = MICE_NAMES[index]
    print 'Mouse %s' % name
    
    N_S = m.data.shape[0]
    # Correlation matrix
    m.corr = np.array([np.corrcoef(np.mean(m.data[i,:,:,:], axis=2))
                       for i in range(N_S)])
    
    # Hierarchical clustering to rearrange the correlation matrix.
    avg_response_timeseries = np.mean(m.data, axis=3)
    dists = [(pdist(i, 'correlation') / 2.0).clip(0.0, 1.0)
             for i in avg_response_timeseries]
    m.linkages = [hac.linkage(dist, method='single', metric='correlation')
                  for dist in dists]
    m.hac_idxs = [hac.dendrogram(Z)['leaves'] for Z in m.linkages]
    hac_response = np.array([m.data[i,idxs,:,:]
                             for i, idxs in enumerate(m.hac_idxs)])
    m.corr_hac = \
            np.array([np.corrcoef(np.mean(hac_response[i,:,:,:],axis=2))
                      for i in range(N_S)])

    ############################################################################
    # Plotting
    # Correlation matrices
    if not os.path.isdir(os.path.join(PLOTS_DIR,'clustering','corr-%s' % name)):
        os.makedirs(os.path.join(PLOTS_DIR, 'clustering', 'corr-%s' % name))

    for d, dirn in enumerate(ORIENTATIONS):
        rows = 1
        cols = 2

        fig = plt.figure(figsize=(cols*5, rows*5))
        fig.suptitle('Mouse %s - %.1f degrees' % (name, dirn))

        sp = fig.add_subplot(rows, cols, 1)
        sp.set_title('Original Correlation Matrix')
        plt.imshow(m.corr[d], vmin=-1, vmax=1, cmap='bwr',
                   interpolation='nearest')
        plt.colorbar(orientation='horizontal', ticks=[-1.0, 0.0, 1.0])
        plt.axis('off')

        sp = fig.add_subplot(rows, cols, 2)
        sp.set_title('Correlation Matrix after HAC')
        plt.imshow(m.corr_hac[d], vmin=-1, vmax=1, cmap='bwr',
                   interpolation='nearest')
        plt.colorbar(orientation='horizontal', ticks=[-1.0, 0.0, 1.0])
        plt.axis('off')

        fig.savefig(os.path.join(PLOTS_DIR,
                                 'clustering/corr-%s/%d.eps' % (name, d)),
                    bbox_inches='tight')
        plt.close()
