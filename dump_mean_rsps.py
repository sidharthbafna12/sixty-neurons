#!/usr/bin/env python
""" dump_mean_rsp.py
"""

import numpy as np
import os

from copy import deepcopy
from src.response import Response
from scipy.signal import decimate

def load_responses(exp_type):
    if exp_type == 'grating':
        from src.params.grating.datafile_params import DATA_DIR, MICE_NAMES
        data_locs = [os.path.join(DATA_DIR,'%s_dir.npy'%c) for c in MICE_NAMES]
        data = map(lambda (n, loc): Response(n, loc), zip(MICE_NAMES,data_locs))
    elif exp_type == 'natural':
        from src.params.naturalmovies.datafile_params import DATA_DIR
        data_locs = [os.path.join(DATA_DIR, '%d.npy' % i) for i in range(11)]
        data = [Response(str(i), data_locs[i]) for i in range(11)]

    return data

def train_test_split(responses, split_type,
                     train_frac=None, to_leave_out=None):
    # Split the responses into a training and a test set.
    # Either we take a fixed proportion of trials from all movie responses, or
    # we leave out the response to one movie entirely.
    # If the index of the movie to be left out is higher than the number of
    # movies seen by the mouse, then we leave out the last movie instead of the
    # number specified.
    N_S, N_TR = responses.data.shape[0], responses.data.shape[3]
    train_rsps = deepcopy(responses)
    test_rsps = deepcopy(responses)
    if split_type == 'even':
        train_rsps.data = train_rsps.data[:,:,:,:int(train_frac*N_TR)]
        test_rsps.data = test_rsps.data[:,:,:,int(train_frac*N_TR):]
    elif split_type == 'loo': # leave-one-out
        if to_leave_out >= N_S:
            print 'Can\'t leave out movie %d as N_S is %d, dropping %d '\
                  'instead.' % (to_leave_out, N_S, N_S-1)
            to_leave_out = N_S - 1
        s_range = np.array(range(to_leave_out) + range(to_leave_out+1,N_S))
        train_rsps.data = train_rsps.data[s_range,:,:,:]
        test_rsps.data = test_rsps.data[to_leave_out:to_leave_out+1,:,:,:]

    return train_rsps, test_rsps

def decimated_responses(rsps, q):
    dec_rsps = deepcopy(rsps)
    dec_rsps.data = decimate(dec_rsps.data, q, axis=2)
    return dec_rsps

exp_type = 'natural'
split_type = 'even'
time_downsample_factor = 5
outdir = './output/mean_rsps/'

responses = load_responses(exp_type)
train_test_splits = map(lambda r: train_test_split(r, split_type,
                                                   train_frac=0.7,
                                                   to_leave_out=0),

                        responses)

for i, response in enumerate(responses):
    name = response.name
    print 'Mouse %s' % name
    
    print 'Splitting out training and test data...'
    tr_rsp, val_rsp = train_test_splits[i]
    
    tr_rsp = decimated_responses(tr_rsp, time_downsample_factor)
    mean_rsps = [tr_rsp.data[:,j,:,:].mean()
                 for j in range(tr_rsp.data.shape[1])]
    outfilepath = os.path.join(outdir, str(i))
    np.save(outfilepath, mean_rsps)
