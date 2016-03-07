# Data file parameters
# Where data is supposed to be, how to find/get it, etc.

import os

MOVIE_DIR = 'data/NaturalMov_Rel'
MOVIE_LOCS = [os.path.join(MOVIE_DIR, 'mov%d.mat' % i)
              for i in range(1,6)]
RSP_BASE_DIR = 'data/data-nov-2015/'
RSP_DIRS = [os.path.join(RSP_BASE_DIR, exp_date, exp_no)
            for exp_date in filter(os.path.isdir, os.listdir(RSP_BASE_DIR))
            for exp_no in filter(os.path.isdir,
                                 os.listdir(os.path.join(RSP_BASE_DIR,
                                                         exp_date)))]
RSP_LOCS = [os.path.join(p, 'AmpMov.mat') for p in RSP_DIRS]
PLOTS_DIR = 'plots/NaturalMovies'
