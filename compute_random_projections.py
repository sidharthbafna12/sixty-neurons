#!/usr/bin/env python
""" compute_random_projections.py
    Computes random projections of stimulus windows in a given video. Window
    length is specified in n_lag.

    Used as a comparison with simulated outputs from the multilayer perceptron
    model fitted to neural response recordings.
"""

import numpy as np
import os

from compute_simulated_responses import dump_features, read_videos

n_neurons = [49, 49, 42, 38, 60, 79, 46, 26, 48, 47, 37]

def transform(rp, video):
    n_in, n_out = rp.shape
    ly, lx, T = video.shape
    n_lag = n_in / (ly * lx)

    padded_video = np.pad(video, ((0,0), (0,0), (n_lag-1,0)), mode='constant')
    X = np.empty((T, n_in))
    for t in range(T):
        X[t,:] = padded_video[:,:,t:t+n_lag].flatten()
    Y = np.dot(X, rp)
    return Y

def compute_rp():
    stim_type = 'breakfast'
    spatial_downsample_factor = 4
    n_lag = 6
    LY, LX = 256 / spatial_downsample_factor, 256 / spatial_downsample_factor

    if stim_type == 'breakfast':
        video_dir = './temp/breakfast_sorted/cam01/'
        features_dir = './temp/video_features_rp/'
        rps_dir = os.path.join(features_dir, 'rps')
        if not os.path.isdir(rps_dir):
            os.makedirs(rps_dir)
    else:
        raise NotImplementedError
    
    print 'Reading in the videos...'
    videos = read_videos(video_dir,
            spatial_downsample_factor=spatial_downsample_factor)
    print 'Read.'

    action_names = os.listdir(video_dir)
    print 'Classes : ', action_names

    
    for i in range(11):
        print 'Random projection %d...' % i
        rp = np.random.standard_normal((LY * LX * n_lag, n_neurons[i]))

        sim_rsps = [[transform(rp, video) for video in class_videos]
                    for class_videos in videos]

        print 'Storing random projections of %s videos...' % stim_type
        output_dir = os.path.join(features_dir, 'rp_%d' % i)
        dump_features(output_dir, sim_rsps, action_names)

        np.save(os.path.join(rps_dir, 'rp_%d' % i), rp)
        print 'Stored.\n'

if __name__ == "__main__":
    compute_rp()
