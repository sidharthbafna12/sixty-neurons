#!/usr/bin/env python
""" reconstruction.py
    Applies linear models in an attempt to reconstruct video stimulus from
    recorded V1 responses. 
    
    The 'reverse' models work with a sliding window applied to the responses,
    with the length specified in n_lag. The model outputs the reconstruced video
    frame corresponding to the window of responses.

    'Forward' models apply the window on the movie, and map a movie window to a
    snapshot of neural responses. The mapping is then inverted to get the movie
    windows from all snapshots of responses, and from there the reconstructed
    movies.

    Training and test data are split in two ways:
        - Test movie has no responses in training set. (Leave-one-out : 'loo')
        - A set fraction of the responses to each movie are taken as the
          training set (even split : 'even')

    Spatially downsampled versions of the movies are used for faster computation
    and to partly compensate for the paucity of data. Clustering of the neural
    responses and regularisation of the filters obtained are also things to
    consider, perhaps.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
import os, time
from distutils.spawn import find_executable

from copy import deepcopy

from src.io import load_responses, load_movies
from src.response import Response
from src.linear_reconstruction import LinearReconstruction
from src.data_manip_utils import train_test_split, smooth_responses

def dump_reconstruction(pred_stim, act_stim, ssims, basedir):
    assert len(pred_stim) == len(act_stim)
    n_movies = len(pred_stim)
    for i in range(n_movies):
        num_trials = len(pred_stim[i])
        for i_tr in range(num_trials):
            dir_path = os.path.join(basedir, 'movie-%s' % str(i),
                                    'trial-%s' % str(i_tr))
            dir_path_comp = os.path.join(dir_path, 'comparison')
            dir_path_orig = os.path.join(dir_path, 'original')
            dir_path_reco = os.path.join(dir_path, 'reconstructed')
            if not os.path.isdir(dir_path):
                os.makedirs(dir_path)
                os.makedirs(dir_path_comp)
                os.makedirs(dir_path_orig)
                os.makedirs(dir_path_reco)

            left_movie = pred_stim[i][i_tr]
            right_movie = act_stim[i]
            T = left_movie.shape[2]
            for t in range(T):
                l_img = left_movie[:,:,t]
                r_img = right_movie[:,:,t]

                # Showing comparison between predicted and actual.
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

                fig.savefig(os.path.join(dir_path_comp, '%d.png' % t))
                plt.close()

                # Saving them individually as well.
                # Reconstructed
                fig = plt.figure()
                fig.set_size_inches(2, 1.5)
                plt.axis('off')

                sp = fig.add_subplot(1, 1, 1)
                plt.imshow(l_img, cmap='gray', interpolation='nearest')
                plt.colorbar(orientation='horizontal',
                             ticks=[l_img.min(), l_img.max()])
                plt.axis('off')

                fig.savefig(os.path.join(dir_path_reco, '%d.eps' % t))
                plt.close()
                
                # Actual
                fig = plt.figure()
                fig.set_size_inches(2, 1.5)
                plt.axis('off')

                sp = fig.add_subplot(1, 1, 1)
                plt.imshow(r_img, cmap='gray', interpolation='nearest')
                plt.colorbar(orientation='horizontal',
                             ticks=[r_img.min(), r_img.max()])
                plt.axis('off')

                fig.savefig(os.path.join(dir_path_orig, '%d.eps' % t))
                plt.close()

            # dump video as well
            if find_executable('ffmpeg'):
                command = 'ffmpeg -framerate %d -i %s/%%d.png %s/video.mp4'\
                        % (CA_SAMPLING_RATE, dir_path_comp, dir_path)
            elif find_executable('avconv'):
                command = 'avconv -framerate %d -i %s/%%d.png %s/video.mp4'\
                        % (CA_SAMPLING_RATE, dir_path_comp, dir_path)
            else:
                print 'No video writer that I know of...'

            print command
            os.system(command)
            time.sleep(0.5)

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
downsample_factor = 8
n_lag = 13
n_clusters = 4
n_components = 16
split_type = 'loo'
model_name = 'linear-regression'
model_type = 'reverse'
regularisation = None

responses = map(smooth_responses, load_responses(exp_type))
movies = load_movies(exp_type, movie_type, downsample_factor=downsample_factor)

for i, response in enumerate(responses):
    if i > 0:
        break

    name = response.name
    print 'Mouse %s' % name
    
    print 'Splitting out training and test data...'
    tr_rsp, te_rsp, tr_mov, te_mov = train_test_split(response, movies,
                                                      split_type,
                                                      train_frac=0.7,
                                                      to_leave_out=0)

    print 'Fitting model parameters for %s %s...' % (model_type, model_name)
    if model_name == 'cca':
        model = LinearReconstruction('cca', model_type, n_lag=n_lag,
                                     n_clusters=n_clusters,
                                     n_components = n_components)
    elif model_name == 'linear-regression':
        model = LinearReconstruction('linear-regression', model_type,
                                     n_clusters=n_clusters, n_lag=n_lag,
                                     regularisation=regularisation)

    model.fit(tr_rsp, tr_mov)

    print 'Reconstructing stimulus movies...'
    te_rsp.data = te_rsp.data[:,:,:,:1]
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
