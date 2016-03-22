#!/usr/bin/python

# Experiment/plotting/other parameters
from src.params.grating.datafile_params import *
from src.params.grating.stimulus_params import *

# Basics
import numpy as np
from matplotlib import pyplot as plt
import os, sys

# GratingResponse tuning properties
from src.double_gaussian_fit import wrapped_double_gaussian
from src.double_gaussian_fit import fit_wrapped_double_gaussian
from src.osi import selectivity_index, pref_direction

# Reading in the data
import scipy.io as sio
from src.grating_response import GratingResponse

data = map(lambda L: GratingResponse(sio.loadmat(L, struct_as_record=False,
                                                 squeeze_me=True)['Data']),
           DATA_LOCS)
ori = map(lambda L:sio.loadmat(L,struct_as_record=False,squeeze_me=True)['Ori'],
          ORI_LOCS)
dirs_rad = np.radians(DIRECTIONS)
sigma0 = 2 * np.pi / len(DIRECTIONS) # initial tuning curve width

for index, m in enumerate(data):
    name = MICE_NAMES[index]
    print 'Mouse %c' % name

    # Get average response over all trials, time
    m.avg_response_dir = np.mean(m.response_dir, axis=(GratingResponse.TrialAxis,
                                                       GratingResponse.TimeAxis))

    # Obtain tuning curves
    init_thetas = [dirs_rad[np.argmax(m.avg_response_dir[:,i])] % np.pi
                   for i in range(m.N)]
    init_cs = np.min(m.avg_response_dir, axis=0)
    init_ws = np.max(m.avg_response_dir, axis=0) \
            - np.min(m.avg_response_dir, axis=0)
    init_sigmas = np.ones(m.N) * sigma0
    init_params = zip(init_thetas, init_sigmas, init_cs, init_ws)
    m.dg_fit_params, m.dg_fit_r2 = \
            zip(*[fit_wrapped_double_gaussian(dirs_rad, m.avg_response_dir[:,i],
                                              p0 = init_params[i])
                  for i in range(m.N)])

    # Find OSI/DSI
    m.osi = selectivity_index(m.response_ori, orientation_flag=True)
    m.pref_orientation = pref_direction(m.response_ori, True)
    m.dsi = selectivity_index(m.response_dir, orientation_flag=False)
    m.pref_direction = pref_direction(m.response_dir, orientation_flag=False)
    
    ############################################################################
    # Plotting
    # Average response
    if not os.path.isdir(os.path.join(PLOTS_DIR,'OrientationTuning')):
        os.makedirs(os.path.join(PLOTS_DIR, 'OrientationTuning'))

    rows = 1; cols = 2
    fig = plt.figure(figsize=(cols*7, rows*4))
    normalised_avg_rsp = m.avg_response_dir / np.sum(m.avg_response_dir, axis=0)

    sp = fig.add_subplot(rows, cols, 1)
    sp.set_title('Average response')
    plt.imshow(m.avg_response_dir, cmap='gray', interpolation='nearest')
    plt.xlabel('Neuron index')
    plt.ylabel('Direction (divided by 22.5 degrees)')

    sp = fig.add_subplot(rows, cols, 2)
    sp.set_title('Normalised Average response')
    plt.imshow(normalised_avg_rsp, cmap='gray', interpolation='nearest')
    plt.xlabel('Neuron index')
    plt.ylabel('Direction (divided by 22.5 degrees)')

    fig.savefig(os.path.join(PLOTS_DIR,'OrientationTuning/AvgRsp-%c.eps'% name),
                bbox_inches='tight')
    plt.close()
    
    # OSI/DSI
    rows = 2; cols = 3
    fig = plt.figure(figsize=(cols*5, rows*5))

    sp = fig.add_subplot(rows, cols, 1)
    sp.set_title('Double Gaussian Quality of Fit (R^2)')
    plt.plot(m.dg_fit_r2)

    sp = fig.add_subplot(rows, cols, 2)
    sp.set_title('OSI')
    plt.plot(m.osi)

    sp = fig.add_subplot(rows, cols, 3)
    sp.set_title('DSI')
    plt.plot(m.dsi)

    sp = fig.add_subplot(rows, cols, 5)
    sp.set_title('Preferred orientation')
    plt.plot(m.pref_orientation)

    sp = fig.add_subplot(rows, cols, 6)
    sp.set_title('Preferred direction')
    plt.plot(m.pref_direction)

    fig.savefig(os.path.join(PLOTS_DIR,
                             'OrientationTuning/Selectivity-%c.eps' % name),
                bbox_inches='tight')
    plt.close()

    # Tuning curves
    if not os.path.isdir(os.path.join(PLOTS_DIR, 'OrientationTuning',
                                      'TuningCurves-%c' % name)):
        os.makedirs(os.path.join(PLOTS_DIR, 'OrientationTuning',
                                 'TuningCurves-%c' % name))
    if not os.path.isdir(os.path.join(PLOTS_DIR, 'OrientationTuning',
                                      'CirVecs-%c' % name)):
        os.makedirs(os.path.join(PLOTS_DIR, 'OrientationTuning',
                                 'CirVecs-%c' % name))

    for i in range(m.N):
        fig, ax = plt.subplots()
        fig.set_size_inches(15, 8)
        fig.suptitle('Mouse %c Neuron %d Orientation Tuning Curve '
                     '(R-squared %.2f)' % (name, i, m.dg_fit_r2[i]))
        plt.xlabel('Stimulus')
        plt.ylabel('Average GratingResponse')
        plt.scatter(DIRECTIONS, m.avg_response_dir[:,i], label='recorded')

        dirs_rad_finer = np.linspace(np.min(dirs_rad),np.max(dirs_rad),num=200)
        plt.plot(np.degrees(dirs_rad_finer),
                 wrapped_double_gaussian(dirs_rad_finer, *m.dg_fit_params[i]),
                 label='double Gaussian fit')
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.85, box.height])
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        fig.savefig(os.path.join(PLOTS_DIR,
                                 'OrientationTuning/TuningCurves-%c/OT-%c_%d.eps'
                                 % (name, name, i)),
                    bbox_inches='tight')
        plt.close()

        # Polar plots showing average response to each orientation.
        fig = plt.figure()
        ax = plt.subplot(111, polar=True)
        ax.plot(dirs_rad,
                m.avg_response_dir[:,i] / np.sum(m.avg_response_dir[:,i]),
                color='r', linewidth=3)
        ax.set_rmax(0.5)
        ax.grid(True)
        ax.set_title('Mouse %c Neuron %d orientation responses' \
                     % (name, i))

        fig.savefig(os.path.join(PLOTS_DIR,
                                 'OrientationTuning/CirVecs-%c/Vecs-%c_%d.eps'
                                 % (name,name,i)),
                    bbox_inches='tight')
        plt.close()
