#!/usr/bin/env python
""" downsample_videos.py
    Downsample the videos generated in make_videos.py using
    scipy.signal.decimate.
    Downsampling factor ranges from 2 to 16 as exponents of 2.
"""

import os

import numpy as np
from scipy.signal import decimate

from src.params.grating.stimulus_params import *
from src.params.naturalmovies.stimulus_params import *

################################################################################
movie_arr_names = ['movie', 'movie_dog', 'movie_ddt', 'movie_dog_ddt']

def load_downsample_and_dump(base_dir, N, movie_name, arr_name):
    MOV_ARR_DIRS= [os.path.join(base_dir, str(i)) for i in range(N)]
    movies = [np.load(os.path.join(d, arr_name + '.npy')) for d in MOV_ARR_DIRS]
    downsampled_movies = [[decimate(decimate(m, 2**p, axis=0, ftype='fir'),
                                    2**p, axis=1, ftype='fir')
                           for m in movies]
                          for p in range(1, 5)]
    for i_p, ms in enumerate(downsampled_movies):
        p = i_p + 1
        print 'Downsampling factor %d:' % 2**p
        for i, m in enumerate(ms):
            print 'Saving %s %s %d...' % (movie_name, arr_name, i)
            if not os.path.isdir(os.path.join(base_dir,str(i),
                                              arr_name+'_down')):
                os.makedirs(os.path.join(base_dir,str(i),arr_name+'_down'))
            np.save(os.path.join(base_dir,str(i),arr_name+'_down/%d'%2**p),m)
            print 'Saved.'

# Reading the grating movies first.
base_dir = './data/grating-movie-video/'
for movie_arr_name in movie_arr_names:
    load_downsample_and_dump(base_dir,len(DIRECTIONS),'grating',movie_arr_name)

"""
################################################################################
# Now the natural scene movies.
base_dir = './data/natural-movie-video-jneuro/'
for movie_arr_name in movie_arr_names:
    load_downsample_and_dump(base_dir, N_MOVIES, 'natural', movie_arr_name)
"""
