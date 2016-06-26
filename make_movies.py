#!/usr/bin/env python
""" make_movies.py
    Create stimulus video arrays, movies, images, etc.

    4 types of movies are output:
        - Original movie resampled to 20 Hz then shown at the CA_SAMPLING_RATE.
        - DoG-filtered version of above movie.
        - First-order differencing applied on the original movie along time
          axis.
        - First-order differencing along time axis on the DoG-filtered movie.

    A window function is applied for the derivative movies to avoid issues at
    the edges.
"""

################################################################################
""" Natural scenes
    ----------------------------------------------------------------------------
    Create natural scene movie videos from mat files.
    Resample the videos to the rate at which the responses are collected (20 Hz)
    as well.
"""

import numpy as np
from matplotlib import pyplot as plt
import os, sys

# Computing DoG
from scipy.ndimage.filters import gaussian_filter as gfilt

# Reading in the data
import scipy.io as sio

# For stimulus movie
import scipy.misc
from scipy.signal import decimate, resample
import matplotlib.animation as mani
from src.io import dump_movie

# DoG radii
r1 = 5
r2 = 2

# Reading in the stimulus movies.
print 'Working on natural scene movies...'

from src.params.naturalmovies.datafile_params import *
from src.params.naturalmovies.stimulus_params import *
# movies_dog = []
for i, p in enumerate(MAT_MOVIE_LOCS):
    print 'Reading movie %d' % i
    # (Y, X, T) convention
    movie= scipy.io.loadmat(p,struct_as_record=False,squeeze_me=True)['movnew']

    # Resample to 20 Hz (MOVIE_REFRESH_RATE / MOVIE_DOWNSAMPLE_FACTOR).
    # Done because Rajeev's J. Neuro paper did it.
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
    
    window = np.dot(np.kaiser(movie.shape[0],2.).reshape((movie.shape[0],1)),
                    np.kaiser(movie.shape[1],2.).reshape((movie.shape[1],1)).T)
    movie_dog=np.dstack([window*(gfilt(movie[:,:,t],r1)-gfilt(movie[:,:,t],r2))
                         for t in range(T)])
    movie_ddt = np.dstack((movie[:,:,0:1], np.diff(movie, axis=2)))
    movie_dog_ddt = np.dstack((movie_dog[:,:,0:1], np.diff(movie_dog, axis=2)))

    print 'Dumping movie as npy/mp4/pngs...'
    dump_movie(os.path.join(MOVIE_DIR, str(i)), movie, CA_SAMPLING_RATE)
    dump_movie(os.path.join(MOVIE_DIR, str(i)), movie_dog, CA_SAMPLING_RATE,
               movie_type='dog')
    dump_movie(os.path.join(MOVIE_DIR, str(i)), movie_ddt, CA_SAMPLING_RATE,
               movie_type='ddt')
    dump_movie(os.path.join(MOVIE_DIR, str(i)), movie_dog_ddt, CA_SAMPLING_RATE,
               movie_type='dog_ddt')

################################################################################
""" Grating scenes
    ----------------------------------------------------------------------------
    Create grating movie videos.
"""

print 'Writing the grating videos...'

# Experiment/plotting/other parameters
from src.params.grating.datafile_params import *
from src.params.grating.stimulus_params import *

# Grating movies
from src.grating import grating_movie

L = GRATING_DURATION * CA_SAMPLING_RATE
movies = [grating_movie(dirn, L) for dirn in DIRECTIONS]

windows = [np.dot(np.kaiser(movie.shape[0],2.).reshape((movie.shape[0],1)),
                  np.kaiser(movie.shape[1],2.).reshape((movie.shape[1],1)).T)
           for movie in movies]
movies_dog = [np.dstack([window*(gfilt(movie[:,:,t],r1)-gfilt(movie[:,:,t],r2))
                         for t in range(L)])
              for window, movie in zip(windows, movies)]
movies_ddt = [np.dstack((movie[:,:,0:1], np.diff(movie, axis=2)))
              for movie in movies]
movies_dog_ddt = [np.dstack((movie[:,:,0:1], np.diff(movie, axis=2)))
                  for movie in movies_dog]

# Time to dump them.
for i, m in enumerate(movies):
    dump_movie(os.path.join(MOVIE_DIR, str(i)), m, CA_SAMPLING_RATE, '')
for i, m in enumerate(movies_dog):
    dump_movie(os.path.join(MOVIE_DIR, str(i)), m, CA_SAMPLING_RATE, 'dog')
for i, m in enumerate(movies_ddt):
    dump_movie(os.path.join(MOVIE_DIR, str(i)), m, CA_SAMPLING_RATE, 'ddt')
for i, m in enumerate(movies_dog_ddt):
    dump_movie(os.path.join(MOVIE_DIR, str(i)), m, CA_SAMPLING_RATE, 'dog_ddt')
