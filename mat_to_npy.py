#!/usr/bin/env python
""" mat_to_npy.py
    Reads in the V1 response data stored as .mat structs of different forms and
    collects them into a single fixed NumPy array format
        (S, N, L, R) : (Stimulus index, Neuron index, Time index, Trial index)
"""

import os
import numpy as np
import scipy.io as sio

from src.params.grating.stimulus_params import *

################################################################################
# Reading the grating dataset first.
print 'Reading the grating dataset...'
MAT_DATA_DIR = 'data/original-data/grating-mat/'
NPY_DATA_DIR = 'data/grating-npy/'
if not os.path.isdir(NPY_DATA_DIR):
    os.makedirs(NPY_DATA_DIR)

G_NAMES_1 = ['A', 'B', 'C', 'D', 'E'] # Data struct is a bit different.
G_NAMES_2 = ['F', 'G1', 'G2', 'G3', 'H', 'I', 'J', 'K']

f_names_1 = [os.path.join(MAT_DATA_DIR, n + '.mat') for n in G_NAMES_1]
f_names_2 = [os.path.join(MAT_DATA_DIR, n + '.mat') for n in G_NAMES_2]

data_1 = [sio.loadmat(loc, struct_as_record=False, squeeze_me=True)['Data']
          for loc in f_names_1]
data_2 = [sio.loadmat(loc, struct_as_record=False, squeeze_me=True)['Ori']
          for loc in f_names_2]

# Just organise the stimulus direction-wise before storing as .npy
L0_full = CA_SAMPLING_RATE * (GRATING_DURATION + GRAY_SCREEN_TIME)
L0 = CA_SAMPLING_RATE * GRATING_DURATION

# For the first 5 mice.
for name, data in zip(G_NAMES_1, data_1):
    print 'Saving %s...' % name
    response = data.Spks
    N, L = response.shape
    slices = np.split(response, L / L0_full, axis=1)
    collected_slices = [[sl[:,-L0:] for (index, sl) in enumerate(slices)
                                    if data.StimSeq[index] == s]
                        for s in DIRECTIONS]

    # collected_slices is an (S, R, N, L)-shaped thing. See src.response.
    # I want (S, N, L, R).
    array = np.array(collected_slices).swapaxes(1, 2).swapaxes(2, 3)
    np.save(os.path.join(NPY_DATA_DIR, name + '_dir'), array)

    # We have stored responses to each direction so far.
    # Average opposite directions to obtain the response to each _orientation_.
    lower, upper = np.split(array, 2, axis=0)
    array_ori = 0.5 * (lower + upper)
    np.save(os.path.join(NPY_DATA_DIR, name + '_ori'), array_ori)

# Now for the rest of them.
for name, data in zip(G_NAMES_2, data_2):
    print 'Saving %s...' % name
    response = data.SpkResponse # (N, (S, (L, R)))
    N = response.shape[0]

    slices = [[response[n][s][-L0:,:] for n in range(N)]
              for s in range(len(DIRECTIONS))]
    np.save(os.path.join(NPY_DATA_DIR, name + '_dir'), np.array(slices))

    lower, upper = np.split(np.array(slices), 2, axis=0)
    array_ori = 0.5 * (lower + upper)
    np.save(os.path.join(NPY_DATA_DIR, name + '_ori'), array_ori)


################################################################################
# Now the natural movies dataset.
print 'Reading the natural movies dataset...'
MAT_DATA_DIR = 'data/original-data/natural-mat/'
NPY_DATA_DIR = 'data/natural-npy/'
if not os.path.isdir(NPY_DATA_DIR):
    os.makedirs(NPY_DATA_DIR)

N_NAMES = ['B', 'D', 'E', 'F', 'G2', 'G3', 'I']
f_names = [os.path.join(MAT_DATA_DIR, n + '.mat') for n in N_NAMES]

data = [sio.loadmat(loc, struct_as_record=False, squeeze_me=True)['Mov']
        for loc in f_names]

for name, d in zip(N_NAMES, data):
    print 'Saving %s...' % name
    response = d.MT_Nat

    array = np.array([[s[:,:,i] for i in range(d.numCells)]
                      for s in response]).swapaxes(2, 3)
    np.save(os.path.join(NPY_DATA_DIR, name), array)

################################################################################
# What's that? Another natural movie dataset?
print 'Reading the second natural movie dataset...'
import re
date_pattern = re.compile('[0-9]{4}-[0-9]{2}-[0-9]{2}')
base_dir = 'data/original-data/data-nov-2015/'
exp_dates = [name for name in os.listdir(base_dir)
             if date_pattern.match(name)]
MAT_DATA_DIRS = [os.path.join(base_dir,d,exp_no) for d in exp_dates
                 for exp_no in os.listdir(os.path.join(base_dir, d))
                 if exp_no in ['1', '2', '3', '4', '5']] # exp_no is a number
NPY_DATA_DIR = 'data/natural-npy-jneuro'
if not os.path.isdir(NPY_DATA_DIR):
    os.makedirs(NPY_DATA_DIR)

f_names = [os.path.join(p, 'AmpMov.mat') for p in MAT_DATA_DIRS]
data = [sio.loadmat(loc, struct_as_record=False, squeeze_me=True)['AmpMov']
        for loc in f_names]

for i, d in enumerate(data):
    print 'Saving %d...' % i
    response = d.MT_nat
    min_trials = min(map(lambda s : s.shape[2], response))
    array = np.array([s[:,:,:min_trials] for s in response])
    np.save(os.path.join(NPY_DATA_DIR, str(i)), array)
