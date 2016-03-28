#!/usr/bin/python

# Basics
import numpy as np
from matplotlib import pyplot as plt
import os, sys

# Experiment/plotting/other parameters
from src.params.grating.datafile_params import *
from src.params.grating.stimulus_params import *
from src.params.grating.pathway_params import *

# Grating movie
from src.grating import grating_movie

# Reading in the data
import scipy.io as sio
from src.grating_response import GratingResponse

################################################################################
print 'Reading in neuron responses...'
data = map(lambda p: GratingResponse(sio.loadmat(p,struct_as_record=False,
                                                 squeeze_me=True)['Data']),
           DATA_LOCS)

NSTA = 5
T = GRATING_DURATION * CA_SAMPLING_RATE
LY, LX = N_PX[1], N_PX[0]

movies = [grating_movie(d, T) for d in DIRECTIONS]
print 'Computing STA...'
for i_m, m in enumerate(data):
    print 'Mouse %d' % i_m

    if not os.path.isdir(os.path.join(PLOTS_DIR, 'Reconstruction', 'STA',
                         'Mouse%d' % i_m)):
        os.makedirs(os.path.join(PLOTS_DIR, 'Reconstruction', 'STA',
                                 'Mouse%d' % i_m))

    # Baseline STA
    baseline_sta = np.zeros((LY,LX,NSTA))
    for i_d in range(len(DIRECTIONS)):
        movie = movies[i_d]
        p_movie = np.pad(movie, ((0,0),(0,0),(NSTA-1,0)), mode='constant')
        # Store sliding window history, then average it.
        sw_history = [p_movie[:,:,ii:ii+NSTA] for ii in range(T)]
        baseline_sta += np.sum(np.array(sw_history), axis=0)
    baseline_sta /= float(len(DIRECTIONS))
    print 'Dumping baseline STA...'
    fig = plt.figure()
    cols = NSTA
    rows = 1
    for i_l in range(1, NSTA+1):
        sp = fig.add_subplot(rows, cols, i_l)
        img = baseline_sta[:,:,i_l-1]
        plt.imshow(img, cmap='gray', interpolation='none')
        plt.colorbar(orientation='horizontal', ticks=[img.min(), img.max()])
        plt.axis('off')
    fig.savefig(os.path.join(PLOTS_DIR,
                             'Reconstruction/STA/Mouse%d/BaseSTA.png'%i_m),
                bbox_inches='tight')
    plt.close()

    for i_n in range(m.N):
        print 'Neuron %d' % i_n
        sta = np.zeros((LY,LX,NSTA))
        for i_d in range(len(DIRECTIONS)):
            print '\tGrating at %f degrees' % DIRECTIONS[i_d]
            movie = movies[i_d]
            p_movie = np.pad(movie, ((0,0),(0,0),(NSTA-1,0)), mode='constant')
            # Store sliding window history, then average it.
            sw_history = [p_movie[:,:,ii:ii+NSTA] for ii in range(T)]
            for i_t in range(NUM_TRIALS):
                print '\t\tTrial %d' % i_t
                sta += np.tensordot(np.array(sw_history),
                                    m.response_dir[i_d,i_t,:,i_n],
                                    axes=(0,0)) \
                       * (m.response_dir[i_d,i_t,:,i_n].sum() / T)
        sta /= float(len(DIRECTIONS) * NUM_TRIALS)

        print '\tDumping STA...'
        fig = plt.figure()
        cols = NSTA
        rows = 1
        for i_l in range(1, NSTA+1):
            sp = fig.add_subplot(rows, cols, i_l)
            img, b_img = sta[:,:,i_l-1], baseline_sta[:,:,i_l-1]
            """
            proj = np.sum(img * b_img) / np.sum(b_img * b_img)
            img -= proj * b_img
            """
            plt.imshow(img, cmap='gray', interpolation='none')
            plt.colorbar(orientation='horizontal', ticks=[img.min(), img.max()])
            plt.axis('off')
        fig.savefig(os.path.join(PLOTS_DIR,
                        'Reconstruction/STA/Mouse%d/Neuron%d.png'%(i_m,i_n)),
                    bbox_inches='tight')
        plt.close()
