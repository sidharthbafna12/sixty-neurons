#!/usr/bin/env python

# Experiment/plotting/other parameters
from src.params.grating.datafile_params import *
from src.params.grating.stimulus_params import *

# Basics
import numpy as np
from matplotlib import pyplot as plt
import os, sys
import scipy

# GratingResponse tuning properties
from src.double_gaussian_fit import wrapped_double_gaussian
from src.double_gaussian_fit import fit_wrapped_double_gaussian
from src.osi import selectivity_index, pref_direction

# Reading in the data
import scipy.io as sio
from src.grating_response import GratingResponse

# Classifiers in action
from src.nn_template import GratingClusterTemplateNN

# ML libraries
from sklearn.metrics import confusion_matrix

data = map(lambda L : GratingResponse(sio.loadmat(L, struct_as_record=False,
                                                  squeeze_me=True)['Data']),
           DATA_LOCS)

n_trials_train = 3
train_data = map(lambda R : R.response_dir[:,:n_trials_train,:,:], data)
test_data = map(lambda R : R.response_dir[:,n_trials_train:,:,:], data)

classifiers = [GratingClusterTemplateNN()]

# Custom loss function total for a confusion matrix. Considers circularity of
# direction.
def loss_fn_matrix(L):
    loss_fn_values = np.zeros(L)
    for i in range(L):
        loss_fn_values[i] = min(i, L-i)
    loss_fn_values /= np.sum(loss_fn_values)
    return scipy.linalg.circulant(loss_fn_values)
    
def conf_mat_badness(mat):
    L = 8
    assert mat.shape == (L, L)
    
    loss_fn_mask = loss_fn_matrix(L)
    return np.sum(loss_fn_mask * mat)

for index, m in enumerate(data):
    name = MICE_NAMES[index]
    print 'Mouse %c' % name

    # Get average response over all trials, time
    m.avg_response_dir = np.mean(m.response_dir, axis=(GratingResponse.TrialAxis,
                                                       GratingResponse.TimeAxis))
    m.avg_response_ori = np.mean(m.response_ori, axis=(GratingResponse.TrialAxis,
                                                       GratingResponse.TimeAxis))

    # Obtain tuning curves
    dirs_rad = np.radians(DIRECTIONS)
    sigma0 = 2 * np.pi / len(DIRECTIONS) # initial tuning curve width
    init_thetas = [dirs_rad[np.argmax(m.avg_response_dir[:,i])] % np.pi
                   for i in range(m.N)]
    init_cs = np.min(m.avg_response_dir, axis=0)
    init_ws = np.max(m.avg_response_dir, axis=0) \
            - np.min(m.avg_response_dir, axis=0)
    init_sigmas = np.ones(m.N) * sigma0
    init_params = zip(init_thetas, init_sigmas, init_cs, init_ws)
    m.dg_fit_params, m.dg_fit_r2 = \
            zip(*[fit_wrapped_double_gaussian(dirs_rad, m.avg_response_dir[:,i],
                                              p0 = init_params[i])
                  for i in range(m.N)])

    # Separate train data to train any classifier.
    m.train = m.response_ori[:,:n_trials_train,:,:]
    m.test = m.response_ori[:,n_trials_train:,:,:]

    m.models = []
    m.predicted_labels = []
    m.conf_mats = []
    m.conf_mat_losses = []
    for c in classifiers:
        c.fit(m.train)
        m.models.append(c)

    correct_dir = np.repeat(np.arange(len(ORIENTATIONS)),
                            NUM_TRIALS - n_trials_train)\
                    .reshape((len(ORIENTATIONS),
                              NUM_TRIALS - n_trials_train))
    for i_c, c in enumerate(m.models):
        pred = c.predict(m.test)
        m.predicted_labels.append(pred)
        conf_mats = [confusion_matrix(correct_dir.flatten(),
                                      pred[:,:,i_n].flatten())
                     for i_n in range(m.N)]
        m.conf_mats.append(conf_mats)
        m.conf_mat_losses.append(map(conf_mat_badness, conf_mats))

        if not os.path.isdir(os.path.join(PLOTS_DIR,
                                          'STClassify',
                                          'Model%d-%c' % (i_c, name))):
            os.makedirs(os.path.join(PLOTS_DIR, 'STClassify',
                                     'Model%d-%c' % (i_c, name)))
        
        for i_n in range(m.N):
            fig = plt.figure()
            plt.imshow(conf_mats[i_n], cmap='gray', interpolation='none')
            plt.colorbar()
            fig.savefig(os.path.join(PLOTS_DIR,
                                     'STClassify/Model%d-%c/ConfMat_%d.eps' \
                                             % (i_c, name, i_n)),
                        bbox_inches='tight')
            plt.close()

    fit_r2_norm = m.dg_fit_r2 - np.mean(m.dg_fit_r2)
    fit_r2_norm /= np.max(np.abs(fit_r2_norm))
    losses = m.conf_mat_losses[0] - np.mean(m.conf_mat_losses[0])
    losses /= np.max(np.abs(losses))
    print 'CC R2 and confmat badness : %1.3f' % np.dot(fit_r2_norm, losses)
