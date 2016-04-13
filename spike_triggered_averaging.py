#!/usr/bin/env python
""" spike_triggered_averaging.py
    Computes the spike triggered average for each neuron.
"""

# Basics
import numpy as np
from matplotlib import pyplot as plt
import os

# Reading in data
from src.response import Response

N_STA = 5
exp_type = 'natural'
movie_type = 'movie_dog_ddt'
downsample_factor = 8

if exp_type == 'grating':
    from src.params.grating.datafile_params import *
    from src.params.grating.stimulus_params import *
elif exp_type == 'natural':
    from src.params.naturalmovies.datafile_params import *
    from src.params.naturalmovies.stimulus_params import *

if exp_type == 'grating':
    data_locs = [os.path.join(DATA_DIR, '%s_dir.npy' % c) for c in MICE_NAMES]
    data = map(lambda (n, loc) : Response(n, loc), zip(MICE_NAMES, data_locs))
elif exp_type == 'natural':
    data_locs = [os.path.join(DATA_DIR, '%d.npy' % i) for i in range(11)]
    data = [Response(str(i), data_locs[i]) for i in range(11)]

if downsample_factor > 1:
    movie_locs = [os.path.join(MOVIE_DIR, str(s), '%s_down' % movie_type,
                               '%d.npy' % downsample_factor)
                  for s in range(N_MOVIES)]
else:
    movie_locs = [os.path.join(MOVIE_DIR, str(s), '%s.npy' % movie_type)
                  for s in range(N_MOVIES)]
movies = map(np.load, movie_locs)
LY, LX = movies[0].shape[:2]

def dumpSTA(sta, out_path, baseline=None):
    fig = plt.figure()
    cols = N_STA
    rows = 1
    for i_l in range(1, N_STA + 1):
        sp = fig.add_subplot(rows, cols, i_l)
        img = sta[:,:,i_l-1]

        if baseline is not None:
            b_img = baseline[:,:,i_l-1]
            proj = np.sum(img * b_img) / np.sum(b_img * b_img)
            img -= proj * b_img

        plt.imshow(img, cmap='gray', interpolation='nearest')
        plt.colorbar(orientation='horizontal', ticks=[img.min(), img.max()])
        plt.axis('off')
    fig.savefig(out_path)
    plt.close()

for index, m in enumerate(data):
    print 'Mouse %s' % m.name
    S, N, T, R = m.data.shape

    # Compute baseline STA for stimuli seen by mouse.
    baseline_sta = np.zeros((LY, LX, N_STA))
    for i_s in range(S):
        movie = movies[i_s]
        p_movie = np.pad(movie, ((0,0), (0,0), (N_STA-1,0)), mode='constant')
        sw_history = [p_movie[:,:,ii:ii+N_STA] for ii in range(T)]
        baseline_sta += np.sum(np.array(sw_history), axis=0)
    baseline_sta /= float(S)

    print 'Dumping baseline STA...'
    basedir = os.path.join(PLOTS_DIR, 'STA', 
                           '%s-D%d' % (movie_type, downsample_factor),
                           'mouse-%s' % m.name)
    if not os.path.isdir(basedir):
        os.makedirs(basedir)
    dumpSTA(baseline_sta, os.path.join(basedir, 'base_sta.png'))

    # Now compute the STA according to the individual cell responses.
    for i_n in range(N):
        print 'Neuron %d' % i_n
        sta = np.zeros((LY, LX, N_STA))
        for i_s in range(S):
            print 'Stimulus %d' % i_s
            movie = movies[i_s]
            p_movie = np.pad(movie, ((0,0), (0,0), (N_STA-1,0)),mode='constant')
            sw_history = [p_movie[:,:,ii:ii+N_STA] for ii in range(T)]
            for i_r in range(R): # trials
                print 'Trial %d' % i_r
                sta += np.tensordot(np.array(sw_history),
                                    m.data[i_s, i_n, :, i_r],
                                    axes=(0,0))\
                     * (m.data[i_s, i_n, :, i_r].sum() / T)
            sta /= float(R)
        sta /= float(S)

        dumpSTA(sta, os.path.join(basedir, 'neuron-%d.png' % i_n), baseline_sta)
