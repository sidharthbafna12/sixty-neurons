#!/usr/bin/env python

""" make_natural_scene_movies.py
    Create natural scene movie videos from mat files.
    Resample the videos to the rate at which the responses are collected (20 Hz)
    as well.
"""

import numpy as np
from matplotlib import pyplot as plt
import os, sys

# Experiment/plotting/other parameters
from src.params.naturalmovies.datafile_params import *
from src.params.naturalmovies.stimulus_params import *

# Reading in the data
import scipy.io as sio

# For stimulus movie
import scipy.misc
from scipy.signal import decimate, resample
import matplotlib.animation as mani

# Reading in the stimulus movies.
movies = []
for i, p in enumerate(MOVIE_LOCS):
    print 'Reading movie %d' % i
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

# Time to dump them.
# First outputting each frame as png or something.
# Can always join them using avconv/ffmpeg later.
VIDEO_OUTPUT_DIR = 'data/stimulus-video'
for i, movie in enumerate(movies):
    print 'Writing movie %d' % i
    if not os.path.isdir(os.path.join(VIDEO_OUTPUT_DIR, '%d' % i, 'frames')):
        os.makedirs(os.path.join(VIDEO_OUTPUT_DIR, '%d' % i, 'frames'))
    
    T = movie.shape[2]
    for t in range(T):
        scipy.misc.toimage(movie[:,:,t]).save(os.path.join(VIDEO_OUTPUT_DIR,
                                                           '%d' % i,
                                                           'frames',
                                                           '%03d.png' % t))

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    img = ax.imshow(movie[:,:,0], cmap='gray', interpolation='nearest')
    
    def update_img(n):
        img.set_data(movie[:,:,n])
        return img

    a = mani.FuncAnimation(fig, update_img, T, repeat=False,
                           interval=1000.0/CA_SAMPLING_RATE)
    writer = mani.writers['ffmpeg'](fps=CA_SAMPLING_RATE)
    a.save(os.path.join(VIDEO_OUTPUT_DIR, '%d' % i, 'movie.mp4'),
           writer=writer, dpi=100)
