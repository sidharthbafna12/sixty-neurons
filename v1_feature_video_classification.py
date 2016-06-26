#!/usr/bin/env python
""" v1_feature_video_classification.py
    Runs a classifier on top of the simulated V1 responses.

    Currently not being used as an HMM-based classifier is implemented in the
    hmm directory. The classifier there uses HTK.
"""

import numpy as np
import os
import cPickle as pickle

# from pyfann import libfann as fann
# from src.hmm_classifier import HMMClassifier
from src.dtw_nn import DTWClassifier

from src.data_manip_utils import confusion_matrix, confusion_matrix_grating
from src.data_manip_utils import cm_goodness

from src.simulated_v1_response import transform, create_mlp_from_params

def read_videos(video_dir, spatial_downsample_factor=4):
    videos = []
    for action in os.listdir(video_dir):
        action_dir = os.path.join(video_dir, action,
                                  'D%d' % spatial_downsample_factor)
        files = (os.path.join(action_dir, f) for f in os.listdir(action_dir))
        yield (np.load(f) for f in files)

def read_grating_videos(dirs):
    for i in range(16):
        movie = np.load(os.path.join(dirs[i], 'movie_down', '4.npy'))
        yield [movie[2:-2,i:i+64,:] for i in range(0,56,5)]

def read_features(features_dir):
    features = []
    for net_index in range(len(os.listdir(features_dir))):
        net_features = []
        net_features_dir = os.path.join(features_dir, 'net%d' % net_index)
        for class_index in os.listdir(net_features_dir):
            class_dir = os.path.join(net_features_dir, class_index)
            files = [os.path.join(class_dir, f) for f in os.listdir(class_dir)]
            net_features.append(map(np.load, files))
        features.append(net_features)
    return features

"""
def confusion_matrix(pred):
    n_classes = len(pred)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for i_s in range(n_classes):
        for prediction in pred[i_s]:
            cm[i_s][prediction] += 1
    return cm

def confusion_matrix_grating(pred,train_classes):
    def remove_all(l, d):
        for i in d:
            l.remove(i)
        return l

    n_classes = 16
    cm = np.zeros((n_classes, n_classes), dtype=int)
    test_classes = remove_all(range(n_classes), train_classes)
    for i, i_te in enumerate(test_classes):
        for p in pred[i]:
            cm[i_te][train_classes[p]] += 1
    return cm
"""

def get_v1_nns(model_dir):
    """
    nns = [fann.neural_net() for i in range(11)]

    paths = ['./output/nets/%d.net' % i for i in range(11)]
    for i, p in enumerate(paths):
        nns[i].create_from_file(p)
    """
    
    nns = []
    for i in range(11):
        with open(os.path.join(model_dir, 'mlp_%d' % i), 'rb') as f:
            Wh, bh, Wo, bo = pickle.load(f)
            nns.append(create_mlp_from_params(Wh, bh, Wo, bo, rectify=True))
    return nns

"""
def transform(net, video, mean_rsp):
    print video.shape
    n_lag = 3

    T = video.shape[2]
    p_video = np.pad(video, ((0,0), (0,0), (n_lag-1,0)), mode='constant')
    
    output = np.zeros((T,net.get_num_output()))
    for t in range(T):
        # print t, '/', T
        output[t,:] = net.run(p_video[:,:,t:t+n_lag].flatten()) + mean_rsp
    output = np.maximum(output, 0)
    return output
"""

def train_test_split(features):
    frac = 0.7
    tr_features = [fs[:int(frac*len(fs))] for fs in features]
    te_features = [fs[int(frac*len(fs)):] for fs in features]
    return tr_features, te_features

def train_test_split_grating(features, tr_idxs):
    tr_features = [features[tr] for tr in tr_idxs]
    te_features = [fs for i, fs in enumerate(features) if i not in tr_idxs]
    return tr_features, te_features

def dump_features(features, f_dir):
    for i, class_features in enumerate(features):
        class_dir = os.path.join(f_dir, str(i))
        os.makedirs(class_dir)
        for j, vector_seq in enumerate(class_features):
            np.save(os.path.join(class_dir, str(j)), class_features[j])

stim_type = 'breakfast'
spatial_downsample_factor = 4
# mean_rsps_dir = './output/mean_rsps/'
nets_dir = './temp/mlp-models'

if stim_type == 'breakfast':
    video_dir = './temp/breakfast_sorted/cam01/'
    features_dir = './temp/video_features'
    predictions_dir = './temp/predictions'
else:
    grating_train_idxs = range(0, 16, 4)
    grating_dirs = [os.path.join('./temp/grating-movie-video', str(i))
                    for i in range(16)]
    features_dir = './temp/video_features_grating'
    predictions_dir = './temp/predictions_grating'

if os.path.isdir(features_dir):
    print 'Reading features from %s...' % features_dir
    features = read_features(features_dir)
else:
    print 'Computing feature vector sequences...'

    """
    print 'Retrieving mean firing rates...'
    mean_rsps = [np.load(os.path.join(mean_rsps_dir, '%d.npy' % p))
                 for p in range(len(os.listdir(mean_rsps_dir)))]
    """

    print 'Retrieving the neural networks...'
    nets = get_v1_nns(nets_dir)
    features = []

    for i, net in enumerate(nets):
        # Read videos.
        # videos is a generator. So have to create it again and again.
        print 'Reading videos...'
        if stim_type == 'grating':
            videos = read_grating_videos(grating_dirs)
        else:
            videos = read_videos(video_dir, spatial_downsample_factor)

        print 'Applying the neural network (%d) transformation...' % i
        """
        net.print_parameters()
        net_features = [[transform(net, video, mean_rsps[i])
                         for video in class_videos]
                        for class_videos in videos]
        """
        net_features = [[transform(net, video)
                         for video in class_videos]
                        for class_videos in videos]

        print 'Dumping the neural features...'
        net_features_dir = os.path.join(features_dir, 'net%d' % i)
        dump_features(net_features, net_features_dir)

        features.append(net_features)

predictions = []
for i, net_features in enumerate(features):
    print 'Using features from network %d...' % i
    if stim_type == 'grating':
        tr_features, te_features = train_test_split_grating(net_features,
                                                            grating_train_idxs)
    else:
        tr_features, te_features = train_test_split(net_features)

    print 'Creating classifier...'
    classifier = DTWClassifier(10)
    classifier.fit(tr_features)

    print 'Prediction from simulated responses...'
    pred = classifier.predict(te_features)
    cm = confusion_matrix(pred)
    # cm = confusion_matrix_grating(pred, grating_train_idxs)
    print cm, cm_goodness(cm, stim_type)
    
    predictions.append(pred)
    pred_outpath = os.path.join(predictions_dir, 'pred_%d.pickle' % i)
    if not os.path.isdir(predictions_dir):
        os.makedirs(predictions_dir)
    with open(pred_outpath, 'w') as pred_out:
        pickle.dump((pred, cm), pred_out)
