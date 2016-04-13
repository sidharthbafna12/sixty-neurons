#!/usr/bin/env python
""" linear_reconstruction.py
    Form the data matrices corresponding to the input-output behaviour shown by
    V1 neurons.
    
    This is done by collecting the response in a window following a given input
    image frame.

    So, f(input) = output
    where input : (L_RSP * N_TRIALS) x (LY * LX)
          output : (L_RSP * N_TRIALS) x (N * N_LAG)

    Assuming a linear relation between the input and the output, we can estimate
    the matrix transforming a given video frame to a series of responses, which
    can be then used to estimate the stimulus from a new set of responses.
"""

import numpy as np
from matplotlib import pyplot as plt
import os

from src.response import Response

exp_type = 'grating'
movie_type = 'movie'
train_frac = 0.5

if exp_type == 'grating':
    from src.params.grating.datafile_params import *
    from src.params.grating.stimulus_params import *
elif exp_type == 'natural':
    from src.params.naturalmovies.datafile_params import *
    from src.params.naturalmovies.stimulus_params import *

if exp_type == 'grating':
    data_locs = [os.path.join(DATA_DIR, '%s_dir.npy' % c) for c in MICE_NAMES]
    data = map(lambda (n, loc) : Response(n, loc), zip(MICE_NAMES, data_locs))
elif exp_type == 'natural':
    data_locs = [os.path.join(DATA_DIR, '%d.npy' % i) for i in range(11)]
    data = [Response(str(i), data_locs[i]) for i in range(11)]

downsample_factor = 8
if downsample_factor > 1:
    movie_locs = [os.path.join(MOVIE_DIR, str(s), '%s_down' % movie_type,
                               '%d.npy' % downsample_factor)
                  for s in range(N_MOVIES)]
else:
    movie_locs = [os.path.join(MOVIE_DIR, str(s), '%s.npy' % movie_type)
                  for s in range(N_MOVIES)]
movies = map(np.load, movie_locs)
LY, LX = movies[0].shape[:2]
movies_flat = map(lambda m : np.vstack([m[:,:,t].flatten()
                                        for t in range(L_RSP)]),
                  movies)

N_LAG = 5 # 5 samples
padded_responses = map(lambda r : np.pad(r.data,((0,0),(0,0),(0,N_LAG-1),(0,0)),
                                         mode='constant'),
                       data)

def get_windows(p_r, trial_range):
    response_windows = []
    n_movies = p_r.shape[0]
    for s in range(n_movies):
        windows = []
        for i_r in trial_range:
            for i_t in range(L_RSP):
                rsp = p_r[s,:,i_t:i_t+N_LAG,i_r].flatten()
                windows.append(rsp)
        response_windows.append(np.vstack(windows))
    return np.array(response_windows)
response_windows_train = \
        map(lambda p:get_windows(p, range(int(train_frac*p.shape[3]))),
            padded_responses)
response_windows_test = \
        map(lambda p:get_windows(p, range(int(train_frac*p.shape[3]),
                                          p.shape[3])),
                          padded_responses)

response_matrices_train = response_windows_train
response_matrices_test = response_windows_test

print 'Linear reconstruction for %s downsampled %d times' \
        % (movie_type, downsample_factor)
for i in range(len(data)):
    m = data[i]
    mat_tr, mat_te = response_matrices_train[i], response_matrices_test[i]
    
    n_tr = m.data.shape[3]
    n_movies = m.data.shape[0]
    movie_matrix_train = \
            np.array(map(lambda m:np.tile(m,(int(train_frac*n_tr),1)),
                     movies_flat[:n_movies]))
    movie_matrix_test =\
            np.array(map(lambda m: np.tile(m, (n_tr - int(train_frac*n_tr),1)),
                     movies_flat[:n_movies]))

    print 'Mouse %s' % m.name
    basedir_path = os.path.join(PLOTS_DIR, 'linear-reconstruction',
                                '%s-D%d-L%d' % (movie_type, downsample_factor,
                                                N_LAG),
                                m.name)

    if not os.path.isdir(basedir_path):
        os.makedirs(basedir_path)

    X_train = np.vstack(mat_tr)
    X_test = np.vstack(mat_te)
    Y_train = np.vstack(movie_matrix_train)
    Y_test = np.vstack(movie_matrix_test)
    
    A, res, rank, sing = np.linalg.lstsq(X_train, Y_train)
    Y_pred = np.dot(X_test, A)

    n_test = Y_test.shape[0]
    for ii in range(n_test):
        if not os.path.isdir(os.path.join(basedir_path, str(int(ii / L_RSP)))):
            os.makedirs(os.path.join(basedir_path, str(int(ii/L_RSP))))
        predicted_img = Y_pred[ii,:].reshape((LY, LX))
        actual_img = Y_test[ii,:].reshape((LY, LX))

        fig = plt.figure()
        sp = fig.add_subplot(1, 2, 1)
        plt.imshow(predicted_img, cmap='gray', interpolation='nearest')
        plt.colorbar(orientation='horizontal', ticks=[predicted_img.min(),
                                                      predicted_img.max()])
        plt.axis('off')
        sp = fig.add_subplot(1, 2, 2)
        plt.imshow(actual_img, cmap='gray', interpolation='nearest')
        plt.colorbar(orientation='horizontal', ticks=[actual_img.min(),
                                                      actual_img.max()])
        plt.axis('off')
        fig.savefig(os.path.join(basedir_path,str(int(ii/L_RSP)),'%d.png'%ii))
        plt.close()
