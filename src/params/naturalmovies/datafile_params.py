# Data file parameters
# Where data is supposed to be, how to find/get it, etc.

import os

MAT_MOVIE_DIR = 'data/original-data/NaturalMov_Rel'
MAT_MOVIE_LOCS = [os.path.join(MOVIE_DIR, 'mov%d.mat' % i)
                  for i in range(1,6)]
DATA_DIR = 'data/natural-npy-jneuro'
MOVIE_DIR = 'data/natural-movie-video-jneuro'
PLOTS_DIR = 'plots/natural-movies'
