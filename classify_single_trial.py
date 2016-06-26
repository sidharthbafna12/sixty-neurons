#!/usr/bin/env python
""" classify_single_trial.py
    Decode stimulus identity from V1 responses on single-trial basis.
    Training data is some fraction out all the trials for each movie for which
    data was recorded. Test data is the remaining.

    Any classifier can be used. I have used nearest-neighbour classifiers here
    with two different distance metrics:
        - Euclidean distance
        - DTW (Dynamic Time Warping) cost function
"""

import numpy as np
from matplotlib import pyplot as plt
import os, time

from copy import deepcopy

from src.response import Response
from src.avg_template import AverageTemplate
from src.dtw_nn import DTWClassifier
from src.clustering import NeuronClustering
from src.correlation import signal_correlation

from src.io import load_responses, load_movies
from src.data_manip_utils import train_test_split, train_test_split_grating
from src.data_manip_utils import confusion_matrix, confusion_matrix_grating
from src.data_manip_utils import cm_goodness
from src.data_manip_utils import smooth_responses

################################################################################
################################################################################
exp_type = 'natural'
movie_type = 'movie'
downsample_factor = 4
split_type = 'even'
n_clusters = 5

responses = map(smooth_responses, load_responses(exp_type))

for r in responses:
    r.clustering = NeuronClustering(n_clusters, signal_correlation)
    r.clustering.fit(r.data)
    r.data = r.clustering.cluster_response(r.data)

# movies = load_movies(exp_type, movie_type, downsample_factor=downsample_factor)
movies = [None] * len(responses)

train_test_splits = map(lambda r: train_test_split(r, movies, split_type,
                                                   train_frac=0.7,
                                                   to_leave_out=0),

                        responses)

for i, response in enumerate(responses):
    name = response.name
    print 'Mouse %s' % name

    print 'Splitting out training and test data...'
    tr_rsp, te_rsp, tr_mov, te_mov = train_test_splits[i]
    # tr_rsp, te_rsp = train_test_splits[i]
    
    print 'Fitting template-matching model...'
    model = AverageTemplate()
    model.fit(tr_rsp)
    print 'Decoding movie indices from test responses...'
    pred_movies = model.predict(te_rsp)
    cm = confusion_matrix(pred_movies)
    # cm = confusion_matrix_grating(pred_movies)
    print cm, cm_goodness(cm, exp_type)
    """
    print 'Fitting DTW model...'
    model = DTWClassifier(3)
    tr_data = [[tr_rsp.data[s,:,:,tr].T for tr in range(tr_rsp.data.shape[3])]
               for s in range(tr_rsp.data.shape[0])]
    model.fit(tr_data)
    te_data = [[te_rsp.data[s,:,:,tr].T for tr in range(te_rsp.data.shape[3])]
               for s in range(te_rsp.data.shape[0])]
    pred_movies = model.predict(te_data)
    cm = confusion_matrix(np.array(pred_movies))
    # cm = confusion_matrix_grating(pred_movies)
    print cm, cm_goodness(cm, exp_type)
    """
