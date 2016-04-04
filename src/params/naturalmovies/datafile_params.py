# Data file parameters
# Where data is supposed to be, how to find/get it, etc.

import os

MOVIE_DIR = 'data/original-data/NaturalMov_Rel'
MOVIE_LOCS = [os.path.join(MOVIE_DIR, 'mov%d.mat' % i)
              for i in range(1,6)]
RSP_BASE_DIR = 'data/natural-npy-jneuro'
PLOTS_DIR = 'plots/natural-movies'
