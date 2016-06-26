#!/usr/bin/env python
""" reliability.py
    Computes response reliability for single neurons.
    Reliability is defined as the pairwise Pearson correlation coefficient
    between different responses to the same movie, averaged over all pairs of
    trials.

    This is but one definition of reliability; there could be other relevant
    ones as well.
"""

import numpy as np
from matplotlib import pyplot as plt
import os

from src.response import Response
from src.reliability import reliability
from src.correlation import signal_correlation, noise_correlation

from src.data_manip_utils import smooth_responses

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

for index in range(len(data)):
    m = data[index]
    print 'Mouse %s' % m.name

    # Smoothing responses. Some unnecessarily clunky stuff here as well.
    # Will do something about that later.
    data[index] = smooth_responses(m)
    m = data[index]
    if exp_type == 'natural':
        m.data = m.data[:5,:,:,:] # Remove K0, K1, etc.
    
    S, N, L, R = m.data.shape
    avg_response = np.mean(m.data, axis=(2,3))

    m.reliability = reliability(m.data)

    # Reliability-tuning curve correlation
    if exp_type == 'grating':
        m.tuning_curve = np.zeros((N,S))
        m.rel_tc_corr = np.zeros(N)
        for i in range(N):
            m.tuning_curve[i,:] = avg_response[:,i]
            r = m.reliability[i,:]
            tc = m.tuning_curve[i,:]
            r -= r.mean()
            tc -= tc.mean()
            m.rel_tc_corr[i] = np.dot(r,tc) / np.sqrt(np.dot(r,r)*np.dot(tc,tc))
    
    ############################################################################
    # Plot
    if not os.path.isdir(os.path.join(PLOTS_DIR, 'reliability',
                                      'mouse-%s' % m.name)):
        os.makedirs(os.path.join(PLOTS_DIR, 'reliability',
                                 'mouse-%s' % m.name))
    
    if exp_type == 'grating':
        fig = plt.figure()
        plt.scatter(np.arange(N), m.rel_tc_corr)
        plt.plot(np.arange(N), m.rel_tc_corr)
        plt.xlabel('Neuron index')
        plt.ylabel('Reliability-tuning curve correlation')
        plt.title('Reliability-tuning curve correlation for all neurons')
        fig.savefig(os.path.join(PLOTS_DIR, 'reliability',
                                 'rel-tc-corr-%s.eps' % m.name),
                    bbox_inches='tight')
        plt.close()

    # Output reliability values.
    for i in range(N):
        fig = plt.figure()
        plt.scatter(np.arange(S), m.reliability[i,:])
        rel, = plt.plot(np.arange(S), m.reliability[i,:], label='reliability')
        handles = [rel]

        if exp_type == 'grating':
            c = m.tuning_curve[i,:]
            c /= c.max()
            c -= c.mean()
            tc, = plt.plot(np.arange(S), c, label='tuning curve')
            handles.append(tc)


        plt.xlabel('%s stimulus index' % exp_type)
        plt.ylabel('Reliability')
        plt.title('Reliability of %s responses' % exp_type)
        plt.legend(handles)
        fig.savefig(os.path.join(PLOTS_DIR, 'reliability',
                                 'mouse-%s' % m.name,
                                 '%d.eps' % i),
                    bbox_inches='tight')
        plt.close()

rel_all = np.hstack([m.reliability.flatten() for m in data]).flatten()
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
fig.savefig(os.path.join(PLOTS_DIR,
                'reliability/reliability_histogram_%s.eps' % exp_type),
            bbox_inches='tight')
plt.close()

if exp_type == 'grating':
    rel_tc_corr_all = np.hstack([m.rel_tc_corr for m in data]).flatten()
    print exp_type, np.median(rel_tc_corr_all), np.std(rel_tc_corr_all)

    # Histogram of rel_tc_corr values.
    n_bins = 50
    fig = plt.figure()
    fig.set_size_inches(3, 2)
    x = np.linspace(np.min(rel_tc_corr_all), np.max(rel_tc_corr_all),
                    num=n_bins+1)[:-1]
    d = (x[-1] - x[0]) / (n_bins-1)
    y = np.histogram(rel_tc_corr_all, bins=n_bins)[0]
    plt.bar(x, y, d)
    plt.locator_params(nbins=3)
    plt.xlabel('Correlation value')
    plt.ylabel('Count')
    plt.title('Correlation value histogram')
    fig.savefig(os.path.join(PLOTS_DIR,
                    'reliability/rel_tc_corr_histogram_%s.eps' % exp_type),
                bbox_inches='tight')
    plt.close()
