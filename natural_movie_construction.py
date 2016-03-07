#!/usr/bin/python

# Trying a forward construction of the neural responses.
# How exactly to do this, will hopefully emerge as I proceed to do this.

# Basics
import numpy as np
from matplotlib import pyplot as plt
import os, sys

# Experiment/plotting/other parameters
from src.params.naturalmovies.datafile_params import *
from src.params.naturalmovies.stimulus_params import *
from src.params.naturalmovies.pathway_params import *
screen = np.zeros(N_PX)

# Reading in the data
import scipy.io as sio

# For stimulus movie
import scipy.misc

# Retinal ganglion cell filters
from src.retina_filter import rgc_filter
from src.thal_filter import thalamus_filter

################################################################################
# Hypothesising some kind of thalamic filterbank here.
# Kind of assuming the thalamic inputs to be the same for all the mice for a
# given movie. This is definitely not the case in reality.
rgc_rsps = []
thal_rsps = []
for i, p in enumerate(NAT_MOVIE_LOCS):
    # Get (X,Y,T) to (T,X,Y) convention...
    movie= scipy.io.loadmat(p,struct_as_record=False,squeeze_me=True)['movnew']\
                .swapaxes(0,2).swapaxes(1,2)
    T = movie.shape[0]
    # Centre about 128, and normalise to [-1,1]
    movie = (movie - 128.0) / 128.0

    # Pass them through generic RGC filters.
    rgc_rf_types = np.random.choice(['on', 'off'],
                                    size=RGC_N_CELLS,
                                    p=[RGC_P, 1 - RGC_P]).T
    rgc_f = rgc_filter(rgc_rf_types)
    rgc_rsp = rgc_f(movie)
    thal_rsp = thalamus_filter(rgc_rsp)

    rgc_rsps.append(rgc_rsp)
    thal_rsps.append(thal_rsp)

    for j in range(T):
        scipy.misc.toimage(movie[j,:,:]).save(os.path.join(PLOTS_DIR,
                                                           'NaturalMovies',
                                                           'Movie%d' % (i+1),
                                                           'frame%03d.png' % j))
        scipy.misc.toimage(rgc_rsp[j,:,:])\
                  .save(os.path.join(PLOTS_DIR,
                                     'NaturalMovies',
                                     'Movie%d' % (i+1),
                                     'rgc_%03d.png' % j))
        scipy.misc.toimage(thal_rsp[j,:,:])\
                  .save(os.path.join(PLOTS_DIR,
                                     'NaturalMovies',
                                     'Movie%d' % (i+1),
                                     'thal_%03d.png' % j))

# Now we can go over the recorded responses...

