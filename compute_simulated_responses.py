#!/usr/bin/env python
""" compute_simulated_responses.py
"""

import numpy as np
import os
import cPickle as pickle

from src.simulated_v1_response import transform, create_mlp_from_params

def get_simulated_responses(model_dir, videos):
    responses = []
    for i in range(11):
        with open(os.path.join(model_dir, 'mlp_%d' % i), 'rb') as f:
            Wh, bh, Wo, bo = pickle.load(f)
            mlp = create_mlp_from_params(Wh, bh, Wo, bo, rectify=True)
            sim_rsps = [[transform(mlp, video) for video in class_videos]
                        for class_videos in videos]
            responses.append(sim_rsps)
    return responses

def dump_features(f_dir, features, names):
    for i, (name, class_features) in enumerate(zip(features, names)):
        class_dir = os.path.join(f_dir, name)
        if not os.path.isdir(class_dir):
            os.makedirs(class_dir)

        for j, vector_seq in enumerate(class_features):
            np.save(os.path.join(class_dir, str(j)), vector_seq)

def read_videos(video_dir, spatial_downsample_factor=4):
    videos = []
    for action in os.listdir(video_dir):
        action_dir = os.path.join(video_dir, action,
                                  'D%d' % spatial_downsample_factor)
        files = (os.path.join(action_dir, f) for f in os.listdir(action_dir))
        videos.append([np.load(f) for f in files])
    return videos

"""
def read_grating_videos(dirs):
    videos = []
    for i in range(16):
        movie = np.load(os.path.join(dirs[i], 'movie_down', '4.npy'))
        videos.append([movie[2:-2,i:i+64,:] for i in range(0,56,5)])
    return videos
"""

def compute_responses():
    stim_type = 'breakfast'
    spatial_downsample_factor = 4
    mlps_dir = './temp/mlp-models/'

    if stim_type == 'breakfast':
        video_dir = './temp/breakfast_sorted/cam01/'
        features_dir = './temp/video_features/'
    else:
        raise NotImplementedError
    
    print 'Reading in the videos...'
    videos = read_videos(video_dir,
            spatial_downsample_factor=spatial_downsample_factor)
    print 'Read.'

    action_names = os.listdir(video_dir)
    print 'Classes : ', action_names

    for i in range(11):
        print 'Using MLP %d...' % i
        with open(os.path.join(mlps_dir, 'mlp_%d' % i), 'rb') as f:
            Wh, bh, Wo, bo = pickle.load(f)
            mlp = create_mlp_from_params(Wh, bh, Wo, bo, rectify=True)
            print 'MLP created from saved parameters. Computing MLP output...'

            sim_rsps = [[transform(mlp, video) for video in class_videos]
                        for class_videos in videos]

            print 'Storing MLP output on %s videos...' % stim_type
            mlp_output_dir = os.path.join(features_dir, 'mlp_%d' % i)
            dump_features(mlp_output_dir, sim_rsps, action_names)
            print 'Stored.\n'

if __name__ == '__main__':
    compute_responses()
