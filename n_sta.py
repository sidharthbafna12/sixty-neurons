#!/usr/bin/python

# Basics
import numpy as np
from matplotlib import pyplot as plt
import os, sys

# Experiment/plotting/other parameters
from src.params.naturalmovies.datafile_params import *
from src.params.naturalmovies.stimulus_params import *
from src.params.naturalmovies.pathway_params import *
screen = np.zeros(N_PX)

# Reading in the data
import scipy.io as sio

# For stimulus movie
import scipy.misc
from scipy.signal import decimate, resample

################################################################################
movies = []

print 'Reading in stimulus movies...'
for i, p in enumerate(MOVIE_LOCS):
    print 'Movie %d' % i
    # (Y, X, T) convention
    movie= scipy.io.loadmat(p,struct_as_record=False,squeeze_me=True)['movnew']

    # Resample to 20 Hz (MOVIE_REFRESH_RATE / MOVIE_DOWNSAMPLE_FACTOR).
    movie_d = movie[:,:,::MOVIE_DOWNSAMPLE_FACTOR]
    movie = np.zeros((movie_d.shape[0], movie_d.shape[1],
                      CA_SAMPLING_RATE * STIMULUS_DURATION))
    # Now upsample back to 50 Hz.
    # Seems intuitive to continue the previous observation...
    for ii in range(CA_SAMPLING_RATE * STIMULUS_DURATION):
        ii_d = int(float(ii)*float(MOVIE_REFRESH_RATE/MOVIE_DOWNSAMPLE_FACTOR)\
                   / CA_SAMPLING_RATE)
        movie[:,:,ii] = movie_d[:,:,ii_d]

    T = movie.shape[2]
    # Centre about 128, and normalise to [-1,1]
    movie = (movie - 128.0) / 128.0
    movies.append(movie)

    if not os.path.isdir(os.path.join(PLOTS_DIR, 'Originals',
                                      'Movie%d' % (i+1))):
        os.makedirs(os.path.join(PLOTS_DIR, 'Originals', 'Movie%d' % (i+1)))

    for j in range(T):
        scipy.misc.toimage(movie[:,:,j]).save(os.path.join(PLOTS_DIR,
                                                           'Originals',
                                                           'Movie%d' % (i+1),
                                                           'frame%03d.png' % j))

################################################################################
# Consider the recorded responses now.
print 'Reading in neuron responses...'
data = map(lambda p: scipy.io.loadmat(p,struct_as_record=False,squeeze_me=True)\
                     ['AmpMov'], RSP_LOCS)

NSTA = 5
LY, LX = movies[0].shape[:2]

print 'Computing STA...'
for i_exp, exp in enumerate(data):
    print 'Experiment number %d' % i_exp
    if not os.path.isdir(os.path.join(PLOTS_DIR, 'Reconstruction', 'STA',
                         'Exp%d' % i_exp)):
        os.makedirs(os.path.join(PLOTS_DIR, 'Reconstruction', 'STA',
                                 'Exp%d' % i_exp))
    M = exp.MT_nat.shape[0]
    N, T, _ = exp.MT_nat[0].shape

    # Baseline STA (constant firing rate)
    baseline_sta = np.zeros((LY, LX, NSTA))
    for i_m in range(M):
        movie = movies[i_m]
        p_movie = np.pad(movie, ((0,0),(0,0),(NSTA-1,0)), mode='constant')
        sw_history = [p_movie[:,:,ii:ii+NSTA] for ii in range(T)]
        baseline_sta += np.sum(np.array(sw_history), axis=0)
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
                             'Reconstruction/STA/Exp%d/BaseSTA.png'%i_exp),
                bbox_inches='tight')
    plt.close()
    
    # Now find this for each neuron
    for i_n in range(N):
        print 'Neuron %d' % i_n
        sta = np.zeros((LY,LX,NSTA))
        for i_m, movie_rsp in enumerate(exp.MT_nat):
            print '\tMovie %d' % i_m
            movie = movies[i_m]
            p_movie = np.pad(movie, ((0,0),(0,0),(NSTA-1,0)), mode='constant')
            # Store sliding window history, then average it.
            sw_history = [p_movie[:,:,ii:ii+NSTA] for ii in range(T)]
            NT = movie_rsp.shape[2]
            for i_t in range(NT):
                print '\t\tTrial %d' % i_t
                sta += np.tensordot(np.array(sw_history),
                                    movie_rsp[i_n,:,i_t],
                                    axes=(0,0)) \
                       * (movie_rsp[i_n,:,i_t].sum() / T)
            # sta /= float(NT)
        # sta /= float(M)

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
                        'Reconstruction/STA/Exp%d/Neuron%d.png'%(i_exp,i_n)),
                    bbox_inches='tight')
        plt.close()
