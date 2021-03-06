#!/usr/bin/env python
""" tuning.py
    Compute orientation tuning curves, orientation selectivity index etc. from
    V1 responses to different sinusoidal grating orientations.
"""

# Experiment/plotting/other parameters
from src.params.grating.datafile_params import *
from src.params.grating.stimulus_params import *

# Basics
import numpy as np

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

import os

# Response tuning properties
from src.gaussian_fit import wrapped_double_gaussian
from src.gaussian_fit import fit_wrapped_double_gaussian
from src.osi import selectivity_index, pref_direction

# Reading in the data
from src.response import Response

# Preprocessing
from src.data_manip_utils import smooth_responses

locs_dirn = [os.path.join(DATA_DIR, '%s_dir.npy' % c) for c in MICE_NAMES]
data_dirn = map(lambda (n, loc): Response(n, loc), zip(MICE_NAMES, locs_dirn))
locs_ori = [os.path.join(DATA_DIR, '%s_ori.npy' % c) for c in MICE_NAMES]
data_ori = map(lambda (n, loc): Response(n, loc), zip(MICE_NAMES, locs_ori))
dirs_rad = np.radians(DIRECTIONS)
sigma0 = 2 * np.pi / len(DIRECTIONS) # initial tuning curve width

for index, (m_dir, m_ori) in enumerate(zip(data_dirn, data_ori)):
    name = MICE_NAMES[index]
    print 'Mouse %s' % name

    # Smooth the responses first.
    m_dir = smooth_responses(m_dir)
    m_ori = smooth_responses(m_ori)
    
    N = m_dir.data.shape[1]

    # Get average response over all trials, time
    m_dir.avg = np.mean(m_dir.data, axis=(2,3))
    m_ori.avg = np.mean(m_ori.data, axis=(2,3))

    # Obtain tuning curves
    init_thetas = [dirs_rad[np.argmax(m_dir.avg[:,i])]%np.pi for i in range(N)]
    init_cs = np.min(m_dir.avg, axis=0)
    init_ws = np.max(m_dir.avg, axis=0) - np.min(m_dir.avg, axis=0)
    init_sigmas = np.ones(N) * sigma0
    init_params = zip(init_thetas, init_sigmas, init_cs, init_ws)
    m_dir.dg_fit_params, m_dir.dg_fit_r2 = \
            zip(*[fit_wrapped_double_gaussian(dirs_rad, m_dir.avg[:,i],
                                              p0 = init_params[i])
                  for i in range(N)])

    # Find OSI/DSI
    m_ori.osi = selectivity_index(m_ori.data, orientation_flag=True)
    m_ori.pref_orientation = pref_direction(m_ori.data, True)
    m_dir.dsi = selectivity_index(m_dir.data, orientation_flag=False)
    m_dir.pref_direction = pref_direction(m_dir.data, orientation_flag=False)
        
    ############################################################################
    # Plotting
    # Average response
    if not os.path.isdir(os.path.join(PLOTS_DIR,'orientation-tuning')):
        os.makedirs(os.path.join(PLOTS_DIR, 'orientation-tuning'))

    rows = 1; cols = 2
    fig = plt.figure(figsize=(cols*7, rows*4))
    normalised_avg_rsp = m_dir.avg / np.sum(m_dir.avg, axis=0)

    sp = fig.add_subplot(rows, cols, 1)
    sp.set_title('Average response')
    plt.imshow(m_dir.avg, cmap='gray', interpolation='nearest')
    plt.xlabel('Neuron index')
    plt.ylabel('Direction (divided by 22.5 degrees)')

    sp = fig.add_subplot(rows, cols, 2)
    sp.set_title('Normalised Average response')
    plt.imshow(normalised_avg_rsp, cmap='gray', interpolation='nearest')
    plt.xlabel('Neuron index')
    plt.ylabel('Direction (divided by 22.5 degrees)')

    fig.savefig(os.path.join(PLOTS_DIR,'orientation-tuning/avg-rsp-%s.eps'% name),
                bbox_inches='tight')
    plt.close()
    
    # OSI/DSI/R^2
    # Plotting histograms of these to show how many neurons are selective etc.
    n_bins = 25
    def plot_hist(index, quant_name, plot_name):
        fig = plt.figure()
        lefts = np.linspace(np.min(index), np.max(index), num=n_bins+1)[:-1]
        width = (np.max(index) - np.min(index)) / n_bins
        plt.bar(lefts, np.histogram(index, bins=n_bins)[0], width)
        plt.ylabel('Count')
        plt.xlabel('%s value' % quant_name)
        plt.title('%s histogram' % quant_name)
        fig.savefig(os.path.join(PLOTS_DIR,
                                 'orientation-tuning/hist-%s.eps' % plot_name),
                    bbox_inches='tight')
        plt.close()
    plot_hist(m_dir.dsi, 'DSI', 'dsi-%s' % name)
    plot_hist(m_ori.osi, 'OSI', 'osi-%s' % name)
    plot_hist(m_dir.dg_fit_r2, 'R^2', 'dg_r2-%s' % name)
    plot_hist(m_dir.pref_direction, 'Preferred direction (grating angle)',
              'pref-dir-%s' % name)
    plot_hist(m_ori.pref_orientation, 'Preferred grating orientation',
              'pref-ori-%s' % name)

    # Tuning curves
    if not os.path.isdir(os.path.join(PLOTS_DIR, 'orientation-tuning',
                                      'tuning-curves-%s' % name)):
        os.makedirs(os.path.join(PLOTS_DIR, 'orientation-tuning',
                                 'tuning-curves-%s' % name))
    if not os.path.isdir(os.path.join(PLOTS_DIR, 'orientation-tuning',
                                      'cir-vecs-%s' % name)):
        os.makedirs(os.path.join(PLOTS_DIR, 'orientation-tuning',
                                 'cir-vecs-%s' % name))

    for i in range(N):
        fig, ax = plt.subplots()
        fig.set_size_inches(4, 2)
        fig.suptitle('R-squared for fit : %.2f' % (m_dir.dg_fit_r2[i]))
        plt.xlabel('Stimulus in degrees')
        plt.ylabel('Average Response')
        plt.scatter(DIRECTIONS, m_dir.avg[:,i], label='recorded')
        plt.xticks([DIRECTIONS[0], DIRECTIONS[-1]])
        plt.yticks(np.linspace(m_dir.avg[:,i].min(),m_dir.avg[:,i].max(),num=3))
        
        y_pred = wrapped_double_gaussian(dirs_rad, *m_dir.dg_fit_params[i])
        y_err = np.absolute(m_dir.avg[:,i] - y_pred)
        dirs_rad_finer = np.linspace(np.min(dirs_rad),np.max(dirs_rad),num=200)
        plt.plot(np.degrees(dirs_rad_finer),
                wrapped_double_gaussian(dirs_rad_finer,*m_dir.dg_fit_params[i]),
                label='fit')
        plt.errorbar(DIRECTIONS, y_pred, yerr=y_err, fmt=None, color='b')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.savefig(os.path.join(PLOTS_DIR,
                            'orientation-tuning/tuning-curves-%s/%s_%d.eps'
                            % (name, name, i)),
                    bbox_inches='tight')
        plt.close()

        # Polar plots showing average response to each direction.
        fig = plt.figure()
        fig.set_size_inches(3, 3)
        ax = plt.subplot(111, polar=True)
        ax.plot(dirs_rad,
                m_dir.avg[:,i] / np.sum(m_dir.avg[:,i]),
                color='r', linewidth=3)
        ax.set_rmax(0.5)
        ax.grid(True)
        ax.set_title('Mouse %s Neuron %d responses' \
                     % (name, i))

        fig.savefig(os.path.join(PLOTS_DIR,
                                 'orientation-tuning/cir-vecs-%s/vecs-%s_%d.eps'
                                 % (name,name,i)),
                    bbox_inches='tight')
        plt.close()
