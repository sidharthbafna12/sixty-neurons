# Data file parameters
# Where data is supposed to be, how to find/get it, etc.

import os

MICE_NAMES = ['A', 'B', 'C', 'D', 'E']
DATA_DIR = 'data/SharedNeuralData'
DATA_LOCS = [os.path.join(DATA_DIR,
                          'Mouse-%c/Data-Mouse%c.mat' % (c, c))
             for c in MICE_NAMES]
ORI_LOCS = [os.path.join(DATA_DIR,
                         'Mouse-%c/Solutions/Ori.mat' % c)
            for c in MICE_NAMES]
PLOTS_DIR = 'plots/Grating'
