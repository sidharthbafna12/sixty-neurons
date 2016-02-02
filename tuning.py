#!/usr/bin/env python

# Experiment/plotting/other parameters
from src.params import *

# Basics
import numpy as np
from matplotlib import pyplot as plt
import os, sys

# Response tuning properties
from src.double_gaussian_fit import wrapped_double_gaussian
from src.double_gaussian_fit import fit_wrapped_double_gaussian
from src.osi import selectivity_index, pref_direction

# Reading in the data
import scipy.io as sio
from src.response import Response

data = map(lambda L: Response(sio.loadmat(L, struct_as_record=False,
                                          squeeze_me=True)['Data']),
           DATA_LOCS)
ori = map(lambda L:sio.loadmat(L,struct_as_record=False,squeeze_me=True)['Ori'],
          ORI_LOCS)
dirs_rad = np.radians(DIRECTIONS)
sigma0 = 2 * np.pi / len(DIRECTIONS) # initial tuning curve width

for i, m in enumerate(data):
    name = MICE_NAMES[i]
    print 'Response %c' % name

    # Get average response over all trials, time
    m.avg_response_dir = np.mean(m.response_dir, axis=(Response.TrialAxis,
                                                       Response.TimeAxis))

    # Obtain tuning curves
    init_thetas = [dirs_rad[np.argmax(m.avg_response_dir[:,i])] % np.pi
                   for i in range(m.N)]
    init_cs = np.min(m.avg_response_dir, axis=0)
    init_ws = np.max(m.avg_response_dir, axis=0) \
            - np.min(m.avg_response_dir, axis=0)
    init_sigmas = np.ones(m.N) * sigma0
    init_params = zip(init_thetas, init_sigmas, init_cs, init_ws)
    m.dg_fit_params, m.dg_fit_r2 = \
            zip(*[fit_wrapped_double_gaussian(dirs_rad, m.avg_response_dir[:,i],
                                              p0 = init_params[i])
                  for i in range(m.N)])

    # Find OSI/DSI
    m.osi = selectivity_index(m.response_ori, orientation_flag=True)
    m.pref_orientation = pref_direction(m.response_ori, True)
    m.dsi = selectivity_index(m.response_dir, orientation_flag=False)
    m.pref_direction = pref_direction(m.response_dir, orientation_flag=False)
