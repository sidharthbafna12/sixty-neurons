#!/usr/bin/env python

# Experiment/plotting/other parameters
from src.params import *

# Basics
import numpy as np
from matplotlib import pyplot as plt
import os, sys

# Clustering
from scipy.spatial.distance import pdist
import scipy.cluster.hierarchy as hac

# Reading in the data
import scipy.io as sio
from src.response import Response

data = map(lambda L: Response(sio.loadmat(L, struct_as_record=False,
                                          squeeze_me=True)['Data']),
           DATA_LOCS)

for index, m in enumerate(data):
    name = MICE_NAMES[index]
    print 'Response %c' % name

    # Correlation matrix
    m.corr_ori = np.array([np.corrcoef(np.mean(m.response_ori,
                                               axis=Response.TrialAxis)[i,:,:].T)
                           for i in range(len(ORIENTATIONS))])
    
    # Hierarchical clustering to rearrange the correlation matrix.
    avg_response_timeseries = np.mean(m.response_ori, axis=Response.TrialAxis)
    dists = [(pdist(i.T, 'correlation') / 2.0).clip(0.0, 1.0)
             for i in avg_response_timeseries]
    m.linkages = [hac.linkage(dist, method='single', metric='correlation')
                  for dist in dists]
    m.hac_idxs = [hac.dendrogram(Z)['leaves'] for Z in m.linkages]
    hac_response_ori = np.array([m.response_ori[i][:,:,idxs]
                                 for i, idxs in enumerate(m.hac_idxs)])
    m.corr_ori_hac = \
            np.array([np.corrcoef(np.mean(hac_response_ori,
                                          axis=Response.TrialAxis)[i,:,:].T)
                      for i in range(len(ORIENTATIONS))])

    ############################################################################
    # Plotting
    # Correlation matrices
    if not os.path.isdir(os.path.join(PLOTS_DIR,'Clustering','Corr-%c' % name)):
        os.makedirs(os.path.join(PLOTS_DIR, 'Clustering', 'Corr-%c' % name))

    for d, dirn in enumerate(ORIENTATIONS):
        rows = 1
        cols = 2

        fig = plt.figure(figsize=(cols*5, rows*5))
        fig.suptitle('Mouse %c - %.1f degrees' % (name, dirn))

        sp = fig.add_subplot(rows, cols, 1)
        sp.set_title('Original Correlation Matrix')
        plt.imshow(m.corr_ori[d], vmin=-1, vmax=1, cmap='bwr',
                   interpolation='nearest')
        plt.colorbar(orientation='horizontal', ticks=[-1.0, 0.0, 1.0])
        plt.axis('off')

        sp = fig.add_subplot(rows, cols, 2)
        sp.set_title('Correlation Matrix after HAC')
        plt.imshow(m.corr_ori_hac[d], vmin=-1, vmax=1, cmap='bwr',
                   interpolation='nearest')
        plt.colorbar(orientation='horizontal', ticks=[-1.0, 0.0, 1.0])
        plt.axis('off')

        fig.savefig(os.path.join(PLOTS_DIR,
                                 'Clustering/Corr-%c/%d.eps' % (name, d)),
                    bbox_inches='tight')
        plt.close()
