#!/usr/bin/env python

# Experiment/plotting/other parameters
from src.params.naturalmovies.datafile_params import *
from src.params.naturalmovies.stimulus_params import *

# Basics
import numpy as np
from matplotlib import pyplot as plt
import os, sys
import scipy

# Reading in the data
import scipy.io as sio

# For stimulus movie
import scipy.misc
from scipy.signal import decimate, resample

# Classifiers in action
from src.nn_template import NatMoviesClusterTemplateNN

# ML libraries
from sklearn.metrics import confusion_matrix

data = map(lambda L : sio.loadmat(L, struct_as_record=False,
                                  squeeze_me=True)['AmpMov'],
           RSP_LOCS)

n_trials_train = 3
classifiers = [NatMoviesClusterTemplateNN()]

for index, exp in enumerate(data):
    print 'Experiment number %d' % index

    # Separate train data to train any classifier.
    exp.train = [exp.MT_nat[i_m][:,:,:n_trials_train]
                 for i_m in range(exp.MT_nat.shape[0])]
    exp.test = [exp.MT_nat[i_m][:,:,n_trials_train:]
                for i_m in range(exp.MT_nat.shape[0])]

    exp.models = []
    exp.predicted_labels = []
    exp.conf_mats = []
    exp.conf_mat_goodness = []
    for c in classifiers:
        c.fit(exp.train)
        exp.models.append(c)

    shapes = [s.shape for s in exp.test]
    correct_movies = [i_m for i_m in range(len(shapes))
                      for i_trial in range(shapes[i_m][2])]
    for i_c, c in enumerate(exp.models):
        pred = c.predict(exp.test)
        exp.predicted_labels.append(pred)
        conf_mats = [confusion_matrix(correct_movies, pred[i_n])
                     for i_n in range(exp.NumNeurons)]
        exp.conf_mats.append(conf_mats)
        exp.conf_mat_goodness.append([float(np.trace(m)) / np.sum(m)
                                      for m in conf_mats])

        if not os.path.isdir(os.path.join(PLOTS_DIR,
                                          'STClassify',
                                          'Model%d_Exp%d' % (i_c, index))):
            os.makedirs(os.path.join(PLOTS_DIR, 'STClassify',
                                     'Model%d_Exp%d' % (i_c, index)))
        
        for i_n in range(exp.NumNeurons):
            fig = plt.figure()
            plt.imshow(conf_mats[i_n], cmap='gray', interpolation='none')
            plt.colorbar()
            fig.savefig(os.path.join(PLOTS_DIR,
                                     'STClassify/Model%d_Exp%d/ConfMat_%d.eps'\
                                             % (i_c, index, i_n)),
                        bbox_inches='tight')
            plt.close()
