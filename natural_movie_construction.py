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
from scipy.signal import decimate, resample

# Visual pathway filters
from src.retina_filter import rgc_filter
from src.retina_filter import rgc_filter2
from src.thal_filter import thalamus_filter
from src.v1_filter import v1_best_fit_filter

################################################################################
# Maybe some kind of thalamic filterbank here.
# Kind of assuming the thalamic inputs to be the same for all the mice for a
# given movie. This is definitely not the case in reality.
movies = []
rgc_rsps = []
thal_rsps = []

for i, p in enumerate(MOVIE_LOCS):
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

    """
    # Pass them through generic RGC filters.
    rgc_rf_types = np.random.choice(['on', 'off'],
                                    size=RGC_N_CELLS,
                                    p=[RGC_P, 1 - RGC_P]).T

    rgc_f = rgc_filter(rgc_rf_types, 'natural')
    rgc_rsp = rgc_f(movie)
    thal_rsp = thalamus_filter(rgc_rsp)

    rgc_rsps.append(rgc_rsp)
    thal_rsps.append(thal_rsp)
    """
    
    if not os.path.isdir(os.path.join(PLOTS_DIR, 'Movie%d' % (i+1))):
        os.makedirs(os.path.join(PLOTS_DIR, 'Movie%d' % (i+1)))
        os.makedirs(os.path.join(PLOTS_DIR, 'Movie%d' % (i+1), 'Original'))
        # os.makedirs(os.path.join(PLOTS_DIR, 'Movie%d' % (i+1), 'RGC'))
        # os.makedirs(os.path.join(PLOTS_DIR, 'Movie%d' % (i+1), 'Saliency'))

    for j in range(T):
        scipy.misc.toimage(movie[:,:,j]).save(os.path.join(PLOTS_DIR,
                                                           'Movie%d' % (i+1),
                                                           'Original',
                                                           'frame%03d.png' % j))
        """
        scipy.misc.toimage(rgc_rsp[:,:,j])\
                  .save(os.path.join(PLOTS_DIR,
                                     'Movie%d' % (i+1),
                                     'RGC',
                                     'rgc_%03d.png' % j))
        scipy.misc.toimage(thal_rsp[:,:,j])\
                  .save(os.path.join(PLOTS_DIR,
                                     'Movie%d' % (i+1),
                                     'Saliency',
                                     'thal_%03d.png' % j))
        """

"""
################################################################################
# Consider the recorded responses now.
data = map(lambda p: scipy.io.loadmat(p,struct_as_record=False,squeeze_me=True)\
                     ['AmpMov'].MT_nat, RSP_LOCS)

print 'Fitting V1 STRFs...'
centres, coeffs, sses = [], [], []
for i_exp, exp in enumerate(data):
    print 'Experiment number %d' % i_exp
    n_movies = exp.shape[0]
    n_neurons, n_samples, _ = exp[0].shape
    centres_mov, coeffs_mov, sses_mov = [], [], []
    for i in range(n_neurons):
        print 'Neuron %d' % i
        centres_neuron, coeffs_neuron, sses_neuron = [], [], []
        for i_mov in range(n_movies):
            print 'Movie %d' % i_mov
            ce, co, sse = v1_best_fit_filter(thal_rsps[i_mov],
                                             exp[i_mov].mean(axis=2)[i,:],
                                             'natural')
            centres_neuron.append(ce)
            coeffs_neuron.append(co)
            sses_neuron.append(sse)
        centres_mov.append(centres_neuron)
        coeffs_mov.append(coeffs_neuron)
        sses_mov.append(sses_neuron)
    centres.append(centres_mov)
    coeffs.append(coeffs_mov)
    sses.append(sses_mov)
"""
