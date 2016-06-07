#!/usr/bin/env python
""" classify_single_trial.py
    Decode stimulus identity from V1 responses on single-trial basis.
"""

import numpy as np
from matplotlib import pyplot as plt
import os, time

from copy import deepcopy

from src.response import Response
from src.avg_template import AverageTemplate
from scipy.signal import decimate

from src.io import load_responses, load_movies
from src.data_manip_utils import train_test_split, train_test_split_grating
from src.data_manip_utils import confusion_matrix, confusion_matrix_grating
from src.data_manip_utils import cm_goodness
from src.data_manip_utils import decimated_movies, decimated_responses

################################################################################
################################################################################
exp_type = 'natural'
movie_type = 'movie'
downsample_factor = 4
split_type = 'even'
n_clusters = 3

responses = load_responses(exp_type)
movies = load_movies(exp_type, movie_type, downsample_factor=downsample_factor)

train_test_splits = map(lambda r: train_test_split(r, movies, split_type,
                                                   train_frac=0.7,
                                                   to_leave_out=0),

                        responses)
"""
train_test_splits = map(lambda r: train_test_split_grating(r,[0,4,8,12]),
                        responses)
"""

for i, response in enumerate(responses):
    name = response.name
    print 'Mouse %s' % name
    
    print 'Splitting out training and test data...'
    tr_rsp, te_rsp, tr_mov, te_mov = train_test_splits[i]
    # tr_rsp, te_rsp = train_test_splits[i]
    
    print 'Fitting template-matching model...'
    model = AverageTemplate(n_clusters)
    model.fit(decimated_responses(tr_rsp, 5))

    print 'Decoding movie indices from test responses...'
    pred_movies = model.predict(decimated_responses(te_rsp, 5))
    cm = confusion_matrix(pred_movies)
    # cm = confusion_matrix_grating(pred_movies)
    print cm, cm_goodness(cm, exp_type)
