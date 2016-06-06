#!/usr/bin/env python
""" classify_single_trial.py
    Decode stimulus identity from V1 responses on single-trial basis.
"""

import numpy as np
from matplotlib import pyplot as plt
import os, time

from copy import deepcopy

from src.response import Response
from src.avg_template import AverageTemplate
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

def load_movies(exp_type, movie_type, downsample_factor=1):
    if exp_type == 'grating':
        from src.params.grating.datafile_params import MOVIE_DIR
        from src.params.grating.stimulus_params import N_MOVIES
    elif exp_type == 'natural':
        from src.params.naturalmovies.datafile_params import MOVIE_DIR
        from src.params.naturalmovies.stimulus_params import N_MOVIES

    if downsample_factor > 1:
        movie_locs = [os.path.join(MOVIE_DIR, str(s), '%s_down' % movie_type,
                                   '%d.npy' % downsample_factor)
                      for s in range(N_MOVIES)]
    else:
        movie_locs = [os.path.join(MOVIE_DIR, str(s), '%s.npy' % movie_type)
                      for s in range(N_MOVIES)]
    movies = map(np.load, movie_locs)
    return movies

def train_test_split(responses, movies, split_type,
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
        train_movies = movies[:N_S]
        test_movies = movies[:N_S]
    elif split_type == 'loo': # leave-one-out
        if to_leave_out >= N_S:
            print 'Can\'t leave out movie %d as N_S is %d, dropping %d '\
                  'instead.' % (to_leave_out, N_S, N_S-1)
            to_leave_out = N_S - 1
        s_range = np.array(range(to_leave_out) + range(to_leave_out+1,N_S))
        train_rsps.data = train_rsps.data[s_range,:,:,:]
        test_rsps.data = test_rsps.data[to_leave_out:to_leave_out+1,:,:,:]
        train_movies = [movies[s] for s in s_range]
        test_movies = [movies[to_leave_out]]

    return train_rsps, test_rsps, train_movies, test_movies

def train_test_split_grating(responses, tr_idxs):
    N_S = responses.data.shape[0]
    tr_rsps = deepcopy(responses)
    te_rsps = deepcopy(responses)
    te_idxs = [i for i in range(N_S) if i not in tr_idxs]
    tr_rsps.data = tr_rsps.data[tr_idxs,:,:,:]
    te_rsps.data = te_rsps.data[te_idxs,:,:,:]
    return tr_rsps, te_rsps

def confusion_matrix(pred):
    n_stim, n_trials = pred.shape
    cm = np.zeros((n_stim, n_stim), dtype=int)
    for i_s in range(n_stim):
        for i_tr in range(n_trials):
            predicted = pred[i_s,i_tr]
            actual = i_s
            cm[actual,predicted] += 1
    return cm

def confusion_matrix_grating(pred):
    def remove_all(l, d):
        for i in d:
            l.remove(i)
        return l

    n_classes = 16
    cm = np.zeros((n_classes, n_classes), dtype=int)
    train_classes = [0,4,8,12]
    test_classes = remove_all(range(n_classes), train_classes)
    for i, i_te in enumerate(test_classes):
        for p in pred[i]:
            cm[i_te][train_classes[p]] += 1
    return cm

def cm_goodness(cm, stim_type):
    goodness = 0

    # distance_table = [0,1,2,3,4,5,6,7,8,7,6,5,4,3,2,1]
    distance_table = [0,1,2,3,4,3,2,1,0,1,2,3,4,3,2,1]
    max_dt = max(distance_table)

    if stim_type == 'natural':
        goodness = float(np.diag(cm).sum()) / cm.sum()
    elif stim_type == 'grating':
        L = cm.shape[0]
        for i in range(L):
            for j in range(L):
                # actual = i, predicted = j
                dist = distance_table[abs(j-i)]
                pred = cm[i][j]
                goodness += (pred * (float(max_dt - dist) / max_dt))
        goodness /= np.sum(cm)
    return goodness


def decimated_movies(movies, q):
    return [decimate(m, q, axis=2) for m in movies]

def decimated_responses(rsps, q):
    dec_rsps = deepcopy(rsps)
    dec_rsps.data = decimate(dec_rsps.data, q, axis=2)
    return dec_rsps

################################################################################
################################################################################
exp_type = 'grating'
if exp_type == 'natural':
    from src.params.naturalmovies.datafile_params import PLOTS_DIR
    from src.params.naturalmovies.stimulus_params import CA_SAMPLING_RATE
elif exp_type == 'grating':
    from src.params.grating.datafile_params import PLOTS_DIR
    from src.params.grating.stimulus_params import CA_SAMPLING_RATE

movie_type = 'movie'
downsample_factor = 4
split_type = 'even'
n_clusters = 3

responses = load_responses(exp_type)
movies = load_movies(exp_type, movie_type, downsample_factor=downsample_factor)

"""
train_test_splits = map(lambda r: train_test_split(r, movies, split_type,
                                                   train_frac=0.7,
                                                   to_leave_out=0),

                        responses)
"""
train_test_splits = map(lambda r: train_test_split_grating(r,[0,4,8,12]),
                        responses)

for i, response in enumerate(responses):
    name = response.name
    print 'Mouse %s' % name
    
    print 'Splitting out training and test data...'
    # tr_rsp, te_rsp, tr_mov, te_mov = train_test_splits[i]
    tr_rsp, te_rsp = train_test_splits[i]
    
    print 'Fitting template-matching model...'
    model = AverageTemplate(n_clusters)
    model.fit(decimated_responses(tr_rsp, 5))

    print 'Decoding movie indices from test responses...'
    # pred_movies = model.predict(decimated_responses(te_rsp, 5))
    pred_movies = model.predict2(decimated_responses(te_rsp, 5))
    # cm = confusion_matrix(pred_movies)
    cm = confusion_matrix_grating(pred_movies)
    print cm, cm_goodness(cm, exp_type)
