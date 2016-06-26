#!/usr/bin/python
""" clustering.py
    Cluster the average responses of mouse neurons to different sinusoidal
    grating orientations to see which sets of neurons seem to fire together.

    This proceeds using the method of hierarchical agglomerative clustering
    which reorders the correlation matrix of neuron responses to collect similar
    neurons together. This can be thresholded to produce a flat set of clusters.
"""

# Basics
import numpy as np
from matplotlib import pyplot as plt
import os

# clustering
from scipy.spatial.distance import squareform
import scipy.cluster.hierarchy as hac

from src.io import load_responses
from src.response import Response
from src.reliability import reliability
from src.correlation import signal_correlation, noise_correlation
from src.clustering import NeuronClustering
from src.data_manip_utils import smooth_responses

n_clusters = 3
exp_type = 'natural'
if exp_type == 'natural':
    from src.params.naturalmovies.datafile_params import PLOTS_DIR
elif exp_type == 'grating':
    from src.params.grating.datafile_params import PLOTS_DIR

data = load_responses(exp_type)
for index, m in enumerate(data):
    print 'Mouse %s' % m.name
    if exp_type == 'natural':
        m.data = m.data[:5,:,:,:]

    data[index] = smooth_responses(m)
    m = data[index]
    
    m.clustering = NeuronClustering(n_clusters, signal_correlation)
    m.clustering.fit(m.data)
    m.data = m.clustering.divnorm(m.data)

    S, N, L, R = m.data.shape

    # Computing signal correlation.
    m.signal_corr = signal_correlation(m.data)

    # Computing noise correlation.
    m.noise_corr = noise_correlation(m.data)

    # Hierarchical clustering to rearrange the correlation matrix.
    # For signal correlation.
    dists_sc = squareform(0.5 * (1.0 - m.signal_corr))
    m.linkage_sc = hac.linkage(dists_sc, method='complete')
    m.hac_idxs_sc = hac.dendrogram(m.linkage_sc)['leaves']
    hac_response_sc = m.data[:,m.hac_idxs_sc,:,:]
    m.signal_corr_hac = signal_correlation(hac_response_sc)

    # Now noise correlation.
    dists_nc = squareform(0.5 * (1.0 - m.noise_corr))
    m.linkage_nc = hac.linkage(dists_nc, method='complete')
    m.hac_idxs_nc = hac.dendrogram(m.linkage_nc)['leaves']
    hac_response_nc = m.data[:,m.hac_idxs_nc,:,:]
    m.noise_corr_hac = noise_correlation(hac_response_nc)
    
    # Cluster reliability.
    # From signal correlations.
    m.cl_idxs_sc = hac.fcluster(m.linkage_sc, n_clusters, criterion='maxclust')
    m.sc_cluster_reliability = np.zeros((n_clusters+1,S))
    m.mean_cluster_rsp_sc = np.zeros((S, n_clusters, L, R))
    for i in range(1, n_clusters+1):
        idxs = np.arange(N)
        idxs = idxs[m.cl_idxs_sc == i]
        m.mean_cluster_rsp_sc[:,i-1,:,:] = m.data[:,idxs,:,:].mean(axis=1)
    m.sc_cluster_reliability = reliability(m.mean_cluster_rsp_sc)
    
    # From noise correlations.
    m.cl_idxs_nc = hac.fcluster(m.linkage_nc, n_clusters, criterion='maxclust')
    m.nc_cluster_reliability = np.zeros((n_clusters+1,S))
    m.mean_cluster_rsp_nc = np.zeros((S, n_clusters, L, R))
    for i in range(1, n_clusters+1):
        idxs = np.arange(N)
        idxs = idxs[m.cl_idxs_nc == i]
        m.mean_cluster_rsp_nc[:,i-1,:,:] = m.data[:,idxs,:,:].mean(axis=1)
    m.nc_cluster_reliability = reliability(m.mean_cluster_rsp_nc)

    ############################################################################
    # Plotting
    if not os.path.isdir(os.path.join(PLOTS_DIR,'clustering')):
        os.makedirs(os.path.join(PLOTS_DIR, 'clustering'))
    if not os.path.isdir(os.path.join(PLOTS_DIR, 'clustering',
                                      'reliability', 'mouse-%s' % m.name)):
        os.makedirs(os.path.join(PLOTS_DIR, 'clustering',
                                 'reliability', 'mouse-%s' % m.name))
    
    rows = 2
    cols = 2

    fig = plt.figure(figsize=(cols*5, rows*5))
    fig.suptitle('Mouse %s' % m.name)

    # Correlation matrices
    sp = fig.add_subplot(rows, cols, 1)
    sp.set_title('Signal Correlation Matrix')
    plt.imshow(m.signal_corr, vmin=-1, vmax=1, cmap='bwr',
               interpolation='nearest')
    plt.colorbar(orientation='horizontal', ticks=[-1.0, 0.0, 1.0])
    plt.axis('off')

    sp = fig.add_subplot(rows, cols, 2)
    sp.set_title('Noise Correlation Matrix')
    plt.imshow(m.noise_corr, vmin=-1, vmax=1, cmap='bwr',
               interpolation='nearest')
    plt.colorbar(orientation='horizontal', ticks=[-1.0, 0.0, 1.0])
    plt.axis('off')

    # After HAC.
    sp = fig.add_subplot(rows, cols, 3)
    sp.set_title('Signal Correlation Matrix (after HAC)')
    plt.imshow(m.signal_corr_hac, vmin=-1, vmax=1, cmap='bwr',
               interpolation='nearest')
    plt.colorbar(orientation='horizontal', ticks=[-1.0, 0.0, 1.0])
    plt.axis('off')

    sp = fig.add_subplot(rows, cols, 4)
    sp.set_title('Noise Correlation Matrix (after HAC)')
    plt.imshow(m.noise_corr_hac, vmin=-1, vmax=1, cmap='bwr',
               interpolation='nearest')
    plt.colorbar(orientation='horizontal', ticks=[-1.0, 0.0, 1.0])
    plt.axis('off')

    fig.savefig(os.path.join(PLOTS_DIR,
                             'clustering/corr-%s.eps' % m.name),
                bbox_inches='tight')
    plt.close()

    # Cluster reliabilities.
    for i in range(n_clusters):
        fig, ax = plt.subplots()
        plt.scatter(np.arange(S), m.sc_cluster_reliability[i,:])
        rel_sc, = plt.plot(np.arange(S), m.sc_cluster_reliability[i,:],
                           label='Signal Correlation cluster reliability')
        plt.scatter(np.arange(S), m.nc_cluster_reliability[i,:])
        rel_nc, = plt.plot(np.arange(S), m.nc_cluster_reliability[i,:],
                           label='Noise Correlation cluster reliability')
        handles = [rel_sc, rel_nc]

        plt.xlabel('%s stimulus index' % exp_type)
        plt.ylabel('Reliability')
        plt.title('Reliability of %s responses' % exp_type)
        plt.legend(handles)
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.savefig(os.path.join(PLOTS_DIR, 'clustering',
                                 'reliability', 'mouse-%s' % m.name,
                                 '%d.eps' % i),
                    bbox_inches='tight')
        plt.close()

rel_all= np.hstack([m.sc_cluster_reliability.flatten() for m in data]).flatten()
print exp_type, np.median(rel_all)
# Histogram of reliability values.
n_bins = 50
fig = plt.figure()
fig.set_size_inches(3, 2)
x = np.linspace(np.min(rel_all), np.max(rel_all), num=n_bins+1)[:-1]
d = (x[-1] - x[0]) / (n_bins-1)
y = np.histogram(rel_all, bins=n_bins)[0]
plt.bar(x, y, d)
plt.locator_params(nbins=3)
plt.xlabel('Reliability')
plt.ylabel('Count')
plt.title('Reliability value histogram')
fig.savefig(os.path.join(PLOTS_DIR, 'clustering', 'reliability',
                         'cluster-reliability-histogram_sc-%s.eps'%exp_type),
            bbox_inches='tight')
plt.close()
