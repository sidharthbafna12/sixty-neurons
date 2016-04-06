#!/usr/bin/env python
""" linear_reconstruction.py
    Form the data matrices corresponding to the input-output behaviour shown by
    V1 neurons.
    
    This is done by collecting the response in a window following a given input
    image frame.

    So, f(input) = output
    where input : (L_RSP * N_TRIALS) x (LY * LX)
          output : (L_RSP * N_TRIALS) x (N * N_LAG)

"""

import numpy as np
from matplotlib import pyplot as plt
import os

from src.response import Response
from src.params.grating.datafile_params import *
from src.params.grating.stimulus_params import *

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet

data_locs = [os.path.join(DATA_DIR, '%s_dir.npy' % c) for c in MICE_NAMES]
data = map(lambda (n, loc) : Response(n, loc), zip(MICE_NAMES, data_locs))

reg='l1l2' # regularisation to use in the linear model
if reg == 'none':
    movie_type = 'movie'
else:
    movie_type = 'movie_dog'

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

n_trials_train = int(0.5 * N_TRIALS)
def get_windows(p_r, trial_range):
    response_windows = []
    for s in range(N_MOVIES):
        windows = []
        for i_r in trial_range:
            for i_t in range(L_RSP):
                rsp = p_r[s,:,i_t:i_t+N_LAG,i_r].flatten()
                windows.append(rsp)
        response_windows.append(np.vstack(windows))
    return np.array(response_windows)
response_windows_train = map(lambda p : get_windows(p, range(n_trials_train)),
                             padded_responses)
response_windows_test = map(lambda p : get_windows(p, range(n_trials_train,
                                                            N_TRIALS)),
                            padded_responses)

response_matrices_train = response_windows_train
response_matrices_test = response_windows_train
movie_matrix_train = np.array(map(lambda m : np.tile(m, (n_trials_train,1)),
                              movies_flat))
movie_matrix_test= np.array(map(lambda m:np.tile(m,(N_TRIALS-n_trials_train,1)),
                            movies_flat))

print 'Linear reconstruction for %s downsampled %d times, regularisation %s' \
        % (movie_type, downsample_factor, reg)
for i in range(len(data)):
    m = data[i]
    mat_tr, mat_te = response_matrices_train[i], response_matrices_test[i]

    print 'Mouse %s' % m.name
    basedir_path = os.path.join(PLOTS_DIR, 'linear-reconstruction',
                                'D%d-L%d-%s' % (downsample_factor, N_LAG, reg),
                                m.name)

    if not os.path.isdir(basedir_path):
        os.makedirs(basedir_path)

    X_train = np.vstack(mat_tr)
    X_test = np.vstack(mat_te)
    Y_train = np.vstack(movie_matrix_train)
    Y_test = np.vstack(movie_matrix_test)
    
    if reg == 'none':
        m.model = LinearRegression()
    elif reg == 'l1':
        m.model = Lasso()
    elif reg == 'l2':
        m.model = Ridge()
    elif reg == 'l1l2':
        m.model = ElasticNet()
    else:
        raise NotImplementedError

    m.model.fit(X_train, Y_train)

    Y_pred = m.model.predict(X_test)
    # Y_pred = np.dot(X_test, A)

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
