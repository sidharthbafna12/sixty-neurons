# Data file parameters
# Where data is supposed to be, how to find/get it, etc.

import os

MOVIE_DIR = 'data/NaturalMov_Rel'
MOVIE_LOCS = [os.path.join(MOVIE_DIR, 'mov%d.mat' % i)
              for i in range(1,6)]
RSP_BASE_DIR = 'data/data-nov-2015/'

import re
date_pattern = re.compile('[0-9]{4}-[0-9]{2}-[0-9]{2}')
EXP_DATES = [name for name in os.listdir(RSP_BASE_DIR)
             if date_pattern.match(name)]
RSP_DIRS = [os.path.join(RSP_BASE_DIR, d, exp_no) for d in EXP_DATES
            for exp_no in os.listdir(os.path.join(RSP_BASE_DIR, d))
            if exp_no in ['1', '2', '3', '4', '5']] # exp_no is a number
RSP_LOCS = [os.path.join(p, 'AmpMov.mat') for p in RSP_DIRS]
PLOTS_DIR = 'plots/NaturalMovies'
