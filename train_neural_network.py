#!/usr/bin/env python
""" train_neural_network.py
"""

import numpy as np
from matplotlib import pyplot as plt
import os, time

from copy import deepcopy

from src.io import load_responses, load_movies
from src.data_manip_utils import smooth_responses, train_test_split
from src.response import Response
from src.avg_template import AverageTemplate

from pyfann import libfann as fann

from scipy.signal import decimate

import logging
logging.getLogger().setLevel(logging.INFO)

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
    for i in range(len(Y_val)):
        pred[i,:] = net.run(X_val[i])
    print mean_absolute_error(np.array(Y_val), pred)

    return net

################################################################################
################################################################################
exp_type = 'natural'
movie_type = 'movie'
spatial_downsample_factor = 4
# time_downsample_factor = 5
n_lag = 6
split_type = 'even'
saved_nns_dir = './temp/nets'

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
    
    tr_rsp = smooth_responses(tr_rsp)
    val_rsp = smooth_responses(val_rsp)

    network = fit_NN(tr_mov, tr_rsp, val_mov, val_rsp, n_lag)
    nns.append(network)

    network.save(os.path.join(saved_nns_dir, '%d.net' % i))
