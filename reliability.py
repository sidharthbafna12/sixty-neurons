#!/usr/bin/env python
""" reliability.py
    Computes response reliability for single neurons.
"""

import numpy as np
from matplotlib import pyplot as plt
import os

from src.response import Response
from src.reliability import reliability
from src.correlation import signal_correlation, noise_correlation

exp_type = 'grating'
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

for m in data:
    print 'Mouse %s' % m.name
    
    S, N, L, R = m.data.shape
    avg_response = np.mean(m.data, axis=(2,3))

    m.reliability = reliability(m.data)

    # Plot
    if not os.path.isdir(os.path.join(PLOTS_DIR, 'reliability',
                                      'mouse-%s' % m.name)):
        os.makedirs(os.path.join(PLOTS_DIR, 'reliability',
                                 'mouse-%s' % m.name))
    
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

