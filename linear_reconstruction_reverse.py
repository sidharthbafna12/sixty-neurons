#!/usr/bin/env python
""" linear_reconstruction_reverse.py
    Performs a linear reconstruction in the reverse fashion compared to
    linear_reconstruction.py.

    Where the other algorithm takes the N_LAG samples following a stimulus, and
    attempts to reconstruct the stimulus from the window of responses following
    it, this one attempts to look at it as a window of stimuli combining to
    yield the responses. This means that the regression problem is much more
    underdetermined, which is where perhaps we can look at L1 regularisation and
    such.
"""

import numpy as np
import os
from matplotlib import pyplot as plt

from sklearn.linear_model import LinearRegression, Lasso, Ridge, ElasticNet
from src.response import Response

exp_type = 'natural'
movie_type = 'movie_dog_ddt'
train_frac = 0.5
reg = 'l2'
N_LAG = 5
downsample_factor = 16

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

if downsample_factor > 1:
    movie_locs = [os.path.join(MOVIE_DIR, str(s), '%s_down' % movie_type,
                               '%d.npy' % downsample_factor)
                  for s in range(N_MOVIES)]
else:
    movie_locs = [os.path.join(MOVIE_DIR, str(s), '%s.npy' % movie_type)
                  for s in range(N_MOVIES)]
movies = map(np.load, movie_locs)
LY, LX = movies[0].shape[:2]

padded_movies = map(lambda m : np.pad(m, ((0,0), (0,0), (N_LAG-1,0)),
                                      mode='constant'),
                    movies)
movies_flat = map(lambda m : np.vstack([m[:,:,t].flatten()
                                        for t in range(m.shape[2])]),
                  padded_movies)
movie_windows = map(lambda mf : np.vstack([np.hstack(mf[t:t+N_LAG,:])
                                           for t in range(L_RSP)]),
                    movies_flat)

responses_train = \
        map(lambda r:np.vstack(
                        [r.data[i_s,:,i_t,i_r]
                         for i_s in range(r.data.shape[0])
                         for i_r in range(int(train_frac*r.data.shape[3]))
                         for i_t in range(L_RSP)]),
            data)
responses_test = \
        map(lambda r:np.vstack(
                        [r.data[i_s,:,i_t,i_r]
                         for i_s in range(r.data.shape[0])
                         for i_r in range(int(train_frac*r.data.shape[3]),
                                          r.data.shape[3])
                         for i_t in range(L_RSP)]),
            data)

def flat_window_to_imgs(windows, img_shape):
    T = windows.shape[0]
    n_px = np.prod(img_shape)
    imgs_flat = np.split(windows, N_LAG, axis=1) # Gives N_LAG-len array
                                                 # of 40 x N_PX arrays
    imgs = map(lambda w : map(lambda f : f.reshape(img_shape), w), imgs_flat)

    # Sum the diagonals right to left.
    # -4 -3 -2 -1  0
    # -3 -2 -1  0  1
    # -2 -1  0  1  2
    # -1  0  1  2  3
    # ...
    out_imgs = []
    for t in range(T):
        img = np.zeros(img_shape)
        for tf in range(t, min(t+N_LAG, T)):
            img += imgs[N_LAG - 1 - (tf - t)][tf]
        img /= (tf - t + 1)
        out_imgs.append(img)
    return out_imgs

print 'Linear reconstruction (reverse) for %s downsampled %d times, '\
      'regularisation %s' % (movie_type, downsample_factor, reg)
for i in range(len(data)):
    m = data[i]
    print 'Mouse %s' % m.name

    mat_tr, mat_te = responses_train[i], responses_test[i]

    n_tr = m.data.shape[3]
    n_movies = m.data.shape[0]
    movie_matrix_train = \
            np.array(map(lambda m:np.tile(m,(int(train_frac*n_tr),1)),
                     movie_windows[:n_movies]))
    movie_matrix_test =\
            np.array(map(lambda m: np.tile(m, (n_tr - int(train_frac*n_tr),1)),
                     movie_windows[:n_movies]))

    basedir_path = os.path.join(PLOTS_DIR, 'linear-reconstruction-reverse',
                                '%s-D%d-L%d-%s' % (movie_type,
                                                   downsample_factor,
                                                   N_LAG, reg),
                                m.name)

    if not os.path.isdir(basedir_path):
        os.makedirs(basedir_path)

    X_train = np.vstack(movie_matrix_train)
    X_test = np.vstack(movie_matrix_test)
    Y_train = np.vstack(mat_tr)
    Y_test = np.vstack(mat_te)

    # Estimated transform matrix.
    print 'Estimating transform matrix...'
    A, res, rank, sing = np.linalg.lstsq(X_train, Y_train)
    if reg == 'none':
        model = LinearRegression()
    elif reg == 'l1':
        model = Lasso()
    elif reg == 'l2':
        model = Ridge()
    elif reg == 'l1l2':
        model = ElasticNet()
    else:
        raise NotImplementedError
    
    print 'Estimating stimulus windows...'
    model.fit(A.T, Y_test.T)
    X_pred = model.coef_
    for ii in range(0, X_pred.shape[0], L_RSP):
        if not os.path.isdir(os.path.join(basedir_path, str(int(ii / L_RSP)))):
            os.makedirs(os.path.join(basedir_path, str(int(ii/L_RSP))))
        
        print 'Reconstructing frames for trial number %d' % int(ii/L_RSP)
        predicted_imgs = flat_window_to_imgs(X_pred[ii:ii+L_RSP,:], (LY,LX))
        
        for t in range(L_RSP):
            predicted_img = predicted_imgs[t]
            actual_img = X_test[ii+t,-(LY*LX):].reshape((LY,LX))

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
            fig.savefig(os.path.join(basedir_path,str(int(ii/L_RSP)),
                                     '%d.png'%(ii+t)))
            plt.close()
