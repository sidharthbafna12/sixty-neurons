#!/usr/bin/env python
""" train_mlp_response_model.py
    Fits a multilayer perceptron to the recorded V1 responses.

    The input layer represents a sliding window over the input video.

    The hidden layer is a tanh activation layer.

    The output layer is a rectified linear unit. It represents the responses
    from the neurons studied.

    One MLP is fit per mouse.
"""
import numpy as np

from src.nnet_regression import MLPRegression, shared_dataset, RegressionModel

from src.io import load_responses, load_movies
from src.data_manip_utils import smooth_responses, train_test_split

import cPickle as pickle
import shutil
import os
os.environ['OMP_NUM_THREADS'] = '16'

import theano
theano.config.openmp = True

def window_matrices(rsp, movs, n_lag):
    def movie_window_matrix(mov, n_lag):
        LY, LX, T = mov.shape
        p_mov = np.pad(mov, ((0,0), (0,0), (n_lag-1,0)), mode='constant')
        mat = np.empty((T, LY*LX*n_lag))
        for t in range(T):
            mat[t,:] = p_mov[:,:,t:t+n_lag].flatten()
        return mat

    def response_window_matrix(rsp, n_lag):
        S,N,T,R = rsp.data.shape
        mat = np.empty((T*R*S,N))

        i = 0
        for s in range(S):
            for r in range(R):
                for t in range(T):
                    mat[i,:] = rsp.data[s,:,t,r]
                    i += 1
        return mat

    mov_mats = [movie_window_matrix(m, n_lag) for m in movs]
    rsp_mat = response_window_matrix(rsp, n_lag)
    n_trials = rsp.data.shape[3]
    
    mov_mat = np.vstack([np.vstack([m for i in range(n_trials)])
                         for m in mov_mats])
    
    return shared_dataset((mov_mat, rsp_mat))

def fit_mlp_models(n_hidden=25, learning_rate=0.15, n_epochs=100,
                   batch_size=600, L1_reg=0.0e-9, L2_reg = 0.0e-9):
    exp_type = 'natural'
    movie_type = 'movie'
    spatial_downsample_factor = 4
    n_lag = 13
    saved_models_dir = './temp/mlp-models-%d' % n_hidden
    predicted_responses_dir = './temp/mlp-predicted-responses-%s' % n_hidden
    if not os.path.isdir(saved_models_dir):
        os.makedirs(saved_models_dir)
    else:
        shutil.rmtree(saved_models_dir)
        os.makedirs(saved_models_dir)
    if not os.path.isdir(predicted_responses_dir):
        os.makedirs(predicted_responses_dir)
    else:
        shutil.rmtree(predicted_responses_dir)
        os.makedirs(predicted_responses_dir)

    print '%d hidden layer neurons, %d epochs to train for'%(n_hidden,n_epochs)
    responses = load_responses(exp_type)
    movies = load_movies(exp_type, movie_type,
                         downsample_factor=spatial_downsample_factor)
    mlp_training_errors = []
    
    for i, response in enumerate(responses):
        name = response.name
        print 'Mouse %s' % name

        print 'Splitting out training and test data...'
        tr_rsp, tst_rsp, tr_mov, tst_mov = train_test_split(response, movies,
                                                        'even', train_frac=0.8)

        print 'Splitting out training and validation data...'
        tr_rsp, val_rsp, tr_mov, val_mov = train_test_split(tr_rsp, tr_mov,
                                                        'even', train_frac=0.9)

        tr_rsp = smooth_responses(tr_rsp)
        val_rsp = smooth_responses(val_rsp)
        tst_rsp = smooth_responses(tst_rsp)

        train_set_x, train_set_y = window_matrices(tr_rsp, tr_mov, n_lag)
        valid_set_x, valid_set_y = window_matrices(val_rsp, val_mov, n_lag)
        test_set_x, test_set_y = window_matrices(tst_rsp, tst_mov, n_lag)

        model = RegressionModel(model_name=name, n_hidden=n_hidden,
                                learning_rate=learning_rate,
                                n_epochs=n_epochs, batch_size=batch_size,
                                L1_reg=L1_reg, L2_reg=L2_reg)

        model.setup_with_data([(train_set_x, train_set_y),
                               (valid_set_x, valid_set_y),
                               (test_set_x, test_set_y)])
        test_error = model.train()
        mlp_training_errors.append(test_error)
        
        predicted = model.y_pred()
        np.save(os.path.join(predicted_responses_dir, 'pred_%s' % name),
                predicted)
        with open(os.path.join(saved_models_dir,'mlp_%s' % name), 'wb') as f:
            pickle.dump(model.regression.params, f)

    with open(os.path.join(saved_models_dir, 'train_errors'), 'wb') as f:
        pickle.dump(mlp_training_errors, f)

if __name__ == "__main__":
    fit_mlp_models()

