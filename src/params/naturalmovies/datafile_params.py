# Data file parameters
# Where data is supposed to be, how to find/get it, etc.

import os

MAT_MOVIE_DIR = './data/RFModel/AmpMovies2/'
"""
MAT_MOVIE_LOCS = [os.path.join(MAT_MOVIE_DIR, 'mov%d.mat' % i)
                  for i in range(1,31)]
"""
MAT_MOVIE_LOCS = [os.path.join(MAT_MOVIE_DIR, 'mov%d.mat' % i)
                  for i in range(1,20) + range(25,31)] # 20-25 = K3?
MAT_DATA_BASE_DIR = 'data/data-nov-2015/'
MAT_DATA_BASE_DIR_2 = 'data/natural-mat/' # From the gratings mice
DATA_DIR = 'temp/natural-npy'
DATA_DIR_2 = 'temp/natural-npy-gmice'

MOVIE_DIR = 'temp/natural-movie-video'
PLOTS_DIR = 'plots/natural-movies'
PLOTS_DIR_2 = 'plots/natural-movies-gmice'
