""" data_manip_utils.py
    Collection of utility functions to deal with the data.
"""

import numpy as np
from copy import deepcopy
from scipy.signal import decimate
from scipy.signal.windows import gaussian
from scipy.ndimage import convolve1d

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

def cm_goodness(cm, stim_type='natural'):
    if stim_type != 'grating':
        goodness = float(np.diag(cm).sum()) / cm.sum()
    else:
        goodness = 0
        # distance_table = [0,1,2,3,4,5,6,7,8,7,6,5,4,3,2,1]
        distance_table = [0,1,2,3,4,3,2,1,0,1,2,3,4,3,2,1]
        max_dt = max(distance_table)
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

def smooth_responses(rsps, window_width=50, sigma=0.65):
    # Gaussian smoothing
    # Values taken from Rajeev's code.
    sm_rsps = deepcopy(rsps)
    sm_rsps.data = convolve1d(rsps.data, gaussian(window_width, sigma), axis=2)
    sm_rsps.data = np.maximum(sm_rsps.data, 0)
    return sm_rsps
