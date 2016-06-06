#!/usr/bin/env python
""" make_movies.py
    Create stimulus video arrays, movies, images, etc.

    4 types of movies are output:
        - Original movie resampled to 20 Hz then shown at the CA_SAMPLING_RATE.
        - DoG-filtered version of above movie.
        - First-order differencing applied on the original movie along time
          axis.
        - First-order differencing along time axis on the DoG-filtered movie.
"""

################################################################################
""" Natural scenes
    ----------------------------------------------------------------------------
    Create natural scene movie videos from mat files.
    Resample the videos to the rate at which the responses are collected (20 Hz)
    as well.
"""

print 'Working on natural scene movies...'

import numpy as np
from matplotlib import pyplot as plt
import os, sys

# Experiment/plotting/other parameters
from src.params.naturalmovies.datafile_params import *
from src.params.naturalmovies.stimulus_params import *

# Computing DoG
from scipy.ndimage.filters import gaussian_filter as gfilt

# Reading in the data
import scipy.io as sio

# For stimulus movie
import scipy.misc
from scipy.signal import decimate, resample
import matplotlib.animation as mani

# DoG radii
r1 = 5
r2 = 2

"""
# Reading in the stimulus movies.
movies = []
movies_dog = []
for i, p in enumerate(MAT_MOVIE_LOCS):
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


    window = np.dot(np.kaiser(movie.shape[0],2.).reshape((movie.shape[0],1)),
                    np.kaiser(movie.shape[1],2.).reshape((movie.shape[1],1)).T)
    movie_dog=np.dstack([window*(gfilt(movie[:,:,t],r1)-gfilt(movie[:,:,t],r2))
                         for t in range(T)])
    movies_dog.append(movie_dog)

movies_ddt = [np.dstack((movie[:,:,0:1], np.diff(movie, axis=2)))
              for movie in movies]
movies_dog_ddt = [np.dstack((movie[:,:,0:1], np.diff(movie, axis=2)))
                  for movie in movies_dog]
"""

# Time to dump them.
# First outputting each frame as png or something.
# Then join them using avconv/ffmpeg later.
def dump_movies(VIDEO_OUTPUT_DIR, movies, movies_dog, movies_ddt,
                movies_dog_ddt):
    for i in range(len(movies)):
        movie = movies[i]
        movie_dog = movies_dog[i]
        movie_ddt = movies_ddt[i]
        movie_dog_ddt = movies_dog_ddt[i]

        print 'Writing movie %d' % i
        
        if not os.path.isdir(os.path.join(VIDEO_OUTPUT_DIR, '%d'%i, 'frames')):
            os.makedirs(os.path.join(VIDEO_OUTPUT_DIR, '%d'%i, 'frames'))
            os.makedirs(os.path.join(VIDEO_OUTPUT_DIR, '%d'%i, 'frames_dog'))
            os.makedirs(os.path.join(VIDEO_OUTPUT_DIR, '%d'%i, 'frames_ddt'))
            os.makedirs(os.path.join(VIDEO_OUTPUT_DIR,'%d'%i,'frames_dog_ddt'))
        
        # Save movie array itself.
        np.save(os.path.join(VIDEO_OUTPUT_DIR,'%d'%i,'movie'), movie)
        np.save(os.path.join(VIDEO_OUTPUT_DIR,'%d'%i,'movie_dog'),movie_dog)
        np.save(os.path.join(VIDEO_OUTPUT_DIR,'%d'%i,'movie_ddt'),movie_ddt)
        np.save(os.path.join(VIDEO_OUTPUT_DIR,'%d'%i,'movie_dog_ddt'),
                movie_dog_ddt)

        # Now the frames.
        T = movie.shape[2]
        for t in range(T):
            scipy.misc.toimage(movie[:,:,t]).save(os.path.join(VIDEO_OUTPUT_DIR,
                                                               '%d' % i,
                                                               'frames',
                                                               '%03d.png' % t))
            scipy.misc.toimage(movie_dog[:,:,t]).save(os.path.join(VIDEO_OUTPUT_DIR,
                                                                   '%d' % i,
                                                                   'frames_dog',
                                                                   '%03d.png' % t))
            scipy.misc.toimage(movie_ddt[:,:,t]).save(os.path.join(VIDEO_OUTPUT_DIR,
                                                                   '%d' % i,
                                                                   'frames_ddt',
                                                                   '%03d.png' % t))
            scipy.misc.toimage(movie_dog_ddt[:,:,t])\
                                    .save(os.path.join(VIDEO_OUTPUT_DIR,
                                                       '%d' % i,
                                                       'frames_dog_ddt',
                                                       '%03d.png' % t))


        # And an mp4 for completeness.
        def write_mp4(m, name):
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)
            img = ax.imshow(m[:,:,0], cmap='gray', interpolation='nearest')

            def update_img(n):
                img.set_data(m[:,:,n])
                return img
            
            a = mani.FuncAnimation(fig, update_img, T, repeat=False,
                                   interval=1000.0/CA_SAMPLING_RATE)
            writer = mani.writers['avconv'](fps=CA_SAMPLING_RATE)
            a.save(os.path.join(VIDEO_OUTPUT_DIR, '%d' % i, name),
                   writer=writer, dpi=100)
            plt.close()
        
        write_mp4(movie, 'movie.mp4')
        write_mp4(movie_dog, 'movie_dog.mp4')
        write_mp4(movie_ddt, 'movie_ddt.mp4')
        write_mp4(movie_dog_ddt, 'movie_dog_ddt.mp4')

"""
dump_movies('data/natural-movie-video-jneuro', movies, movies_dog,
            movies_ddt, movies_dog_ddt)
"""

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
dump_movies('data/grating-movie-video', movies, movies_dog, movies_ddt,
            movies_dog_ddt)
