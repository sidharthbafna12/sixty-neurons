#!/usr/bin/env python
""" reconstruction.py
    A more refined version of the reconstruction pipeline implemented in
    linear_reconstruction.py.
"""

import numpy as np
from matplotlib import pyplot as plt
import os, time

from copy import deepcopy

from src.response import Response
from src.reverse_correlation import FlatPriorReverseCorrelation
from src.reverse_correlation import OptimalPriorReverseCorrelation 
from src.linear_reconstruction import LinearReconstruction

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

def dump_reconstruction(pred_stim, act_stim, ssims, basedir):
    assert len(pred_stim) == len(act_stim)
    n_movies = len(pred_stim)
    for i in range(n_movies):
        num_trials = len(pred_stim[i])
        for i_tr in range(num_trials):
            dir_path = os.path.join(basedir, 'movie-%s' % str(i),
                                    'trial-%s' % str(i_tr))
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)

            left_movie = pred_stim[i][i_tr]
            right_movie = act_stim[i]
            T = left_movie.shape[2]
            for t in range(T):
                l_img = left_movie[:,:,t]
                r_img = right_movie[:,:,t]

                fig = plt.figure()
                plt.title('Frame %d (SSIM %.3f)' % (t, ssims[i][i_tr][t]))
                plt.axis('off')

                sp = fig.add_subplot(1, 2, 1)
                plt.imshow(l_img, cmap='gray', interpolation='nearest')
                plt.colorbar(orientation='horizontal',
                             ticks=[l_img.min(), l_img.max()])
                plt.axis('off')

                sp = fig.add_subplot(1, 2, 2)
                plt.imshow(r_img, cmap='gray', interpolation='nearest')
                plt.colorbar(orientation='horizontal',
                             ticks=[r_img.min(), r_img.max()])
                plt.axis('off')

                fig.savefig(os.path.join(dir_path, '%d.png' % t))
                plt.close()

            # dump video as well
            command = 'ffmpeg -framerate %d -i %s/%%d.png %s/video.mp4'\
                    % (CA_SAMPLING_RATE, dir_path, dir_path)
            print command
            os.system(command)
            time.sleep(0.5)

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
downsample_factor = 4
n_lag = 11
n_clusters = 4
n_components = 64
split_type = 'loo'
model_name = 'linear-regression'
model_type = 'reverse'
regularisation = None

responses = load_responses(exp_type)
movies = load_movies(exp_type, movie_type, downsample_factor=downsample_factor)

train_test_splits = map(lambda r: train_test_split(r, movies, split_type,
                                                   train_frac=0.7,
                                                   to_leave_out=0),

                        responses)

for i, response in enumerate(responses):
    name = response.name
    print 'Mouse %s' % name
    
    print 'Splitting out training and test data...'
    tr_rsp, te_rsp, tr_mov, te_mov = train_test_splits[i]

    print 'Fitting model parameters for %s %s...' % (model_type, model_name)
    if model_name == 'flat-prior':
        model = FlatPriorReverseCorrelation(n_lag = n_lag)
    elif model_name == 'optimal-prior':
        model = OptimalPriorReverseCorrelation(n_lag = n_lag)
    elif model_name == 'cca':
        model = LinearReconstruction('cca', model_type, n_lag=n_lag,
                                     n_clusters=n_clusters,
                                     n_components = n_components)
    elif model_name == 'linear-regression':
        model = LinearReconstruction('linear-regression', model_type,
                                     n_clusters=n_clusters, n_lag=n_lag,
                                     regularisation=regularisation)

    model.fit(tr_rsp, tr_mov)

    print 'Reconstructing stimulus movies...'
    pred_movies = model.predict(te_rsp)
    ssims = model.reconstruction_quality(pred_movies, te_mov)

    print 'Dumping reconstructed movies...'
    dump_dir = os.path.join(PLOTS_DIR, 'reconstruction',
                            model_name, model_type, 'split-%s' % split_type,
                            '%s-D%d-L%d'%(movie_type,downsample_factor,n_lag),
                            'mouse-%s' % name)
    if not os.path.isdir(dump_dir):
        os.makedirs(dump_dir)

    dump_reconstruction(pred_movies, te_mov, ssims, dump_dir)
