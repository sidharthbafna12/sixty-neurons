#!/usr/bin/env python
""" single_trial_classify.py
    Classify a single trial V1 response as one of the possible stimulus types as
    described in the training data.

    There is not that much data that one can try particularly complicated
    things, so a nearest-neighbour/template-matching approach is probably the
    most sensible option.
"""

################################################################################
# Experiment/plotting/other parameters.
from src.params.grating.datafile_params import *
# from src.params.naturalmovies.datafile_params import *

# Basics
import numpy as np
from matplotlib import pyplot as plt
import os

# Reading in the data
from src.response import Response

# For classification and for confusion matrix
from src.nn_template import ClusterTemplateNN
from sklearn.metrics import confusion_matrix

data_locs= [os.path.join(DATA_DIR, '%s_ori.npy' % c) for c in MICE_NAMES]
data = map(lambda (n, loc): Response(n, loc), zip(MICE_NAMES, data_locs))
# data_locs= [os.path.join(RSP_BASE_DIR, '%d.npy' % i) for i in range(11)]
# data = map(lambda (i, loc) : Response(str(i), loc), enumerate(data_locs))

n_train = 3
for index, m in enumerate(data):
    name = MICE_NAMES[index]
    print 'Mouse %s' % name
    # print 'Experiment %d' % index
    
    S, N, T, N_TR = m.data.shape
    m.stc_train = m.data[:,:,:,:n_train]
    m.stc_test = m.data[:,:,:,n_train:]
    
    m.stc_model = ClusterTemplateNN(K=10)
    m.stc_model.fit(m.stc_train)
    m.stc_true_labels = np.repeat(np.arange(S), N_TR - n_train)\
                          .reshape((S, N_TR - n_train))
    m.stc_pred_labels = m.stc_model.predict(m.stc_test)
    m.stc_conf_mats = [confusion_matrix(m.stc_true_labels.flatten(),
                                        m.stc_pred_labels[:,:,i_n].flatten())
                       for i_n in range(N)]
