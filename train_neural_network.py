#!/usr/bin/env python
""" train_neural_network.py
"""

import numpy as np
from matplotlib import pyplot as plt
import os, time

from copy import deepcopy

from src.response import Response
from src.avg_template import AverageTemplate

from pyfann import libfann as fann

from scipy.signal import decimate

import logging
logging.getLogger().setLevel(logging.INFO)

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

def decimated_movies(movies, q):
    return [decimate(m, q, axis=2) for m in movies]

def decimated_responses(rsps, q):
    dec_rsps = deepcopy(rsps)
    dec_rsps.data = decimate(dec_rsps.data, q, axis=2)
    return dec_rsps

def fit_NN(tr_movies, tr_responses, val_movies, val_responses, n_lag):
    p_movs = [np.pad(m, ((0,0), (0,0), (n_lag-1,0)), mode='constant')
              for m in tr_movies]
    
    tr_data = tr_responses.data
    mean_rsps = []
    for i in range(tr_data.shape[1]):
        mean_rsps.append(tr_data[:,i,:,:].mean())
        tr_data[:,i,:,:] -= tr_data[:,i,:,:].mean()

    LY, LX, T = tr_movies[0].shape
    S, N, T, R = tr_data.shape

    num_input = LY*LX*n_lag
    num_output = N
    num_hidden = 1000
    epochs = 50
    epochs_between_reports = 1
    desired_error = 0.25
    learning_rate = 0.7
    connection_rate = 0.1

    train_DS = []
    val_DS = []

    for i_s in range(S):
        # Training data
        for i_tr in range(R):
            for i_t in range(T):
                inp_tr = p_movs[i_s][:,:,i_t:i_t+n_lag].flatten()
                out_tr = tr_data[i_s,:,i_t,i_tr]
                train_DS.append([inp_tr, out_tr])

    X_tr, Y_tr = zip(*train_DS)

    p_movs = [np.pad(m, ((0,0), (0,0), (n_lag-1,0)), mode='constant')
              for m in val_movies]
    val_data = val_responses.data
    for i in range(N):
        val_data[:,i,:,:] -= mean_rsps[i]
    S, N, T, R = val_data.shape
    for i_s in range(S):
        # Validation data
        for i_tr in range(R):
            for i_t in range(T):
                inp_val = p_movs[i_s][:,:,i_t:i_t+n_lag].flatten()
                out_val = val_data[i_s,:,i_t,i_tr]
                val_DS.append([inp_val, out_val])
    X_val, Y_val = zip(*val_DS)
    
    train_data = fann.training_data()
    train_data.set_train_data(X_tr, Y_tr)

    net = fann.neural_net()
    net.create_sparse_array(connection_rate, (num_input,num_hidden,num_output))
    net.set_learning_rate(learning_rate)
    net.set_activation_function_output(fann.LINEAR)
    net.set_activation_function_hidden(fann.SIGMOID_SYMMETRIC_STEPWISE)
    net.train_on_data(train_data, epochs, epochs_between_reports, desired_error)

    pred = np.zeros((len(Y_val), N))
    errors = []
    for i in range(len(Y_val)):
        pred[i,:] = net.run(X_val[i])

        err = np.absolute(pred[i,:] - Y_val[i])
        err = err / (np.maximum(np.abs(pred[i,:]), np.abs(Y_val[i]))\
                    + 0.00001)
        errors.append(err)

    print np.median(errors, axis=0)

    return net

################################################################################
################################################################################
exp_type = 'natural'
if exp_type == 'natural':
    from src.params.naturalmovies.datafile_params import PLOTS_DIR
    from src.params.naturalmovies.stimulus_params import CA_SAMPLING_RATE
elif exp_type == 'grating':
    from src.params.grating.datafile_params import PLOTS_DIR
    from src.params.grating.stimulus_params import CA_SAMPLING_RATE

movie_type = 'movie'
spatial_downsample_factor = 4
time_downsample_factor = 5
n_lag = 3
split_type = 'even'

responses = load_responses(exp_type)
movies = load_movies(exp_type, movie_type,
                     downsample_factor=spatial_downsample_factor)
train_test_splits = map(lambda r: train_test_split(r, movies, split_type,
                                                   train_frac=0.7,
                                                   to_leave_out=0),

                        responses)

nns = []
for i, response in enumerate(responses):
    name = response.name
    print 'Mouse %s' % name
    
    print 'Splitting out training and test data...'
    tr_rsp, val_rsp, tr_mov, val_mov = train_test_splits[i]
    
    tr_rsp = decimated_responses(tr_rsp, time_downsample_factor)
    val_rsp = decimated_responses(val_rsp, time_downsample_factor)
    tr_mov = decimated_movies(tr_mov, time_downsample_factor)
    val_mov = decimated_movies(val_mov, time_downsample_factor)
    network = fit_NN(tr_mov, tr_rsp, val_mov, val_rsp, n_lag)
    nns.append(network)

    network.save('./output/nets/%d.net' % i)
